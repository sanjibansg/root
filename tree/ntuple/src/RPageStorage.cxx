/// \file RPageStorage.cxx
/// \ingroup NTuple
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-04

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RPageStorage.hxx>
#include <ROOT/RPageStorageFile.hxx>
#include <ROOT/RColumn.hxx>
#include <ROOT/RFieldBase.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleMetrics.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleSerialize.hxx>
#include <ROOT/RNTupleZip.hxx>
#include <ROOT/RPageAllocator.hxx>
#include <ROOT/RPageSinkBuf.hxx>
#include <ROOT/RPageStorageFile.hxx>
#ifdef R__ENABLE_DAOS
#include <ROOT/RPageStorageDaos.hxx>
#endif

#include <Compression.h>
#include <TError.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstring>
#include <functional>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <utility>

using ROOT::Internal::MakeUninitArray;
using ROOT::Internal::RClusterDescriptorBuilder;
using ROOT::Internal::RClusterGroupDescriptorBuilder;
using ROOT::Internal::RColumn;
using ROOT::Internal::RColumnDescriptorBuilder;
using ROOT::Internal::RColumnElementBase;
using ROOT::Internal::RExtraTypeInfoDescriptorBuilder;
using ROOT::Internal::RFieldDescriptorBuilder;
using ROOT::Internal::RNTupleSerializer;

using ROOT::Internal::RCluster;
using ROOT::Internal::ROnDiskPage;
using ROOT::Internal::ROnDiskPageMap;

using ROOT::Experimental::Detail::RNTupleAtomicCounter;
using ROOT::Experimental::Detail::RNTupleAtomicTimer;
using ROOT::Experimental::Detail::RNTupleCalcPerf;
using ROOT::Experimental::Detail::RNTupleMetrics;
using ROOT::Experimental::Detail::RNTupleTickCounter;

ROOT::Internal::RPageStorage::RPageStorage(std::string_view name)
   : fMetrics(""), fPageAllocator(std::make_unique<ROOT::Internal::RPageAllocatorHeap>()), fNTupleName(name)
{
}

ROOT::Internal::RPageStorage::~RPageStorage() {}

void ROOT::Internal::RPageStorage::RSealedPage::ChecksumIfEnabled()
{
   if (!fHasChecksum)
      return;

   auto charBuf = reinterpret_cast<const unsigned char *>(fBuffer);
   auto checksumBuf = const_cast<unsigned char *>(charBuf) + GetDataSize();
   std::uint64_t xxhash3;
   RNTupleSerializer::SerializeXxHash3(charBuf, GetDataSize(), xxhash3, checksumBuf);
}

ROOT::RResult<void> ROOT::Internal::RPageStorage::RSealedPage::VerifyChecksumIfEnabled() const
{
   if (!fHasChecksum)
      return RResult<void>::Success();

   auto success = RNTupleSerializer::VerifyXxHash3(reinterpret_cast<const unsigned char *>(fBuffer), GetDataSize());
   if (!success)
      return R__FAIL("page checksum verification failed, data corruption detected");
   return RResult<void>::Success();
}

ROOT::RResult<std::uint64_t> ROOT::Internal::RPageStorage::RSealedPage::GetChecksum() const
{
   if (!fHasChecksum)
      return R__FAIL("invalid attempt to extract non-existing page checksum");

   assert(fBufferSize >= kNBytesPageChecksum);
   std::uint64_t checksum;
   RNTupleSerializer::DeserializeUInt64(
      reinterpret_cast<const unsigned char *>(fBuffer) + fBufferSize - kNBytesPageChecksum, checksum);
   return checksum;
}

//------------------------------------------------------------------------------

void ROOT::Internal::RPageSource::RActivePhysicalColumns::Insert(ROOT::DescriptorId_t physicalColumnId,
                                                                 RColumnElementBase::RIdentifier elementId)
{
   auto [itr, _] = fColumnInfos.emplace(physicalColumnId, std::vector<RColumnInfo>());
   for (auto &columnInfo : itr->second) {
      if (columnInfo.fElementId == elementId) {
         columnInfo.fRefCounter++;
         return;
      }
   }
   itr->second.emplace_back(RColumnInfo{elementId, 1});
}

void ROOT::Internal::RPageSource::RActivePhysicalColumns::Erase(ROOT::DescriptorId_t physicalColumnId,
                                                                RColumnElementBase::RIdentifier elementId)
{
   auto itr = fColumnInfos.find(physicalColumnId);
   R__ASSERT(itr != fColumnInfos.end());
   for (std::size_t i = 0; i < itr->second.size(); ++i) {
      if (itr->second[i].fElementId != elementId)
         continue;

      itr->second[i].fRefCounter--;
      if (itr->second[i].fRefCounter == 0) {
         itr->second.erase(itr->second.begin() + i);
         if (itr->second.empty()) {
            fColumnInfos.erase(itr);
         }
      }
      break;
   }
}

ROOT::Internal::RCluster::ColumnSet_t ROOT::Internal::RPageSource::RActivePhysicalColumns::ToColumnSet() const
{
   RCluster::ColumnSet_t result;
   for (const auto &[physicalColumnId, _] : fColumnInfos)
      result.insert(physicalColumnId);
   return result;
}

bool ROOT::Internal::RPageSource::REntryRange::IntersectsWith(const ROOT::RClusterDescriptor &clusterDesc) const
{
   if (fFirstEntry == ROOT::kInvalidNTupleIndex) {
      /// Entry range unset, we assume that the entry range covers the complete source
      return true;
   }

   if (clusterDesc.GetNEntries() == 0)
      return true;
   if ((clusterDesc.GetFirstEntryIndex() + clusterDesc.GetNEntries()) <= fFirstEntry)
      return false;
   if (clusterDesc.GetFirstEntryIndex() >= (fFirstEntry + fNEntries))
      return false;
   return true;
}

ROOT::Internal::RPageSource::RPageSource(std::string_view name, const ROOT::RNTupleReadOptions &options)
   : RPageStorage(name), fOptions(options)
{
}

ROOT::Internal::RPageSource::~RPageSource() {}

std::unique_ptr<ROOT::Internal::RPageSource>
ROOT::Internal::RPageSource::Create(std::string_view ntupleName, std::string_view location,
                                    const ROOT::RNTupleReadOptions &options)
{
   if (ntupleName.empty()) {
      throw RException(R__FAIL("empty RNTuple name"));
   }
   if (location.empty()) {
      throw RException(R__FAIL("empty storage location"));
   }
   if (location.find("daos://") == 0)
#ifdef R__ENABLE_DAOS
      return std::make_unique<ROOT::Experimental::Internal::RPageSourceDaos>(ntupleName, location, options);
#else
      throw RException(R__FAIL("This RNTuple build does not support DAOS."));
#endif

   return std::make_unique<ROOT::Internal::RPageSourceFile>(ntupleName, location, options);
}

ROOT::Internal::RPageStorage::ColumnHandle_t
ROOT::Internal::RPageSource::AddColumn(ROOT::DescriptorId_t fieldId, RColumn &column)
{
   R__ASSERT(fieldId != ROOT::kInvalidDescriptorId);
   auto physicalId =
      GetSharedDescriptorGuard()->FindPhysicalColumnId(fieldId, column.GetIndex(), column.GetRepresentationIndex());
   R__ASSERT(physicalId != ROOT::kInvalidDescriptorId);
   fActivePhysicalColumns.Insert(physicalId, column.GetElement()->GetIdentifier());
   return ColumnHandle_t{physicalId, &column};
}

void ROOT::Internal::RPageSource::DropColumn(ColumnHandle_t columnHandle)
{
   fActivePhysicalColumns.Erase(columnHandle.fPhysicalId, columnHandle.fColumn->GetElement()->GetIdentifier());
}

void ROOT::Internal::RPageSource::SetEntryRange(const REntryRange &range)
{
   if ((range.fFirstEntry + range.fNEntries) > GetNEntries()) {
      throw RException(R__FAIL("invalid entry range"));
   }
   fEntryRange = range;
}

void ROOT::Internal::RPageSource::LoadStructure()
{
   if (!fHasStructure)
      LoadStructureImpl();
   fHasStructure = true;
}

void ROOT::Internal::RPageSource::Attach(RNTupleSerializer::EDescriptorDeserializeMode mode)
{
   LoadStructure();
   if (!fIsAttached)
      GetExclDescriptorGuard().MoveIn(AttachImpl(mode));
   fIsAttached = true;
}

std::unique_ptr<ROOT::Internal::RPageSource> ROOT::Internal::RPageSource::Clone() const
{
   auto clone = CloneImpl();
   if (fIsAttached) {
      clone->GetExclDescriptorGuard().MoveIn(GetSharedDescriptorGuard()->Clone());
      clone->fHasStructure = true;
      clone->fIsAttached = true;
   }
   return clone;
}

ROOT::NTupleSize_t ROOT::Internal::RPageSource::GetNEntries()
{
   return GetSharedDescriptorGuard()->GetNEntries();
}

ROOT::NTupleSize_t ROOT::Internal::RPageSource::GetNElements(ColumnHandle_t columnHandle)
{
   return GetSharedDescriptorGuard()->GetNElements(columnHandle.fPhysicalId);
}

void ROOT::Internal::RPageSource::UnzipCluster(RCluster *cluster)
{
   if (fTaskScheduler)
      UnzipClusterImpl(cluster);
}

void ROOT::Internal::RPageSource::UnzipClusterImpl(RCluster *cluster)
{
   RNTupleAtomicTimer timer(fCounters->fTimeWallUnzip, fCounters->fTimeCpuUnzip);

   const auto clusterId = cluster->GetId();
   auto descriptorGuard = GetSharedDescriptorGuard();
   const auto &clusterDescriptor = descriptorGuard->GetClusterDescriptor(clusterId);

   fPreloadedClusters[clusterDescriptor.GetFirstEntryIndex()] = clusterId;

   std::atomic<bool> foundChecksumFailure{false};

   std::vector<std::unique_ptr<RColumnElementBase>> allElements;
   const auto &columnsInCluster = cluster->GetAvailPhysicalColumns();
   for (const auto columnId : columnsInCluster) {
      // By the time we unzip a cluster, the set of active columns may have already changed wrt. to the moment when
      // we requested reading the cluster. That doesn't matter much, we simply decompress what is now in the list
      // of active columns.
      if (!fActivePhysicalColumns.HasColumnInfos(columnId))
         continue;
      const auto &columnInfos = fActivePhysicalColumns.GetColumnInfos(columnId);

      allElements.reserve(allElements.size() + columnInfos.size());
      for (const auto &info : columnInfos) {
         allElements.emplace_back(GenerateColumnElement(info.fElementId));

         const auto &pageRange = clusterDescriptor.GetPageRange(columnId);
         std::uint64_t pageNo = 0;
         std::uint64_t firstInPage = 0;
         for (const auto &pi : pageRange.GetPageInfos()) {
            auto onDiskPage = cluster->GetOnDiskPage(ROnDiskPage::Key{columnId, pageNo});
            RSealedPage sealedPage;
            sealedPage.SetNElements(pi.GetNElements());
            sealedPage.SetHasChecksum(pi.HasChecksum());
            sealedPage.SetBufferSize(pi.GetLocator().GetNBytesOnStorage() + pi.HasChecksum() * kNBytesPageChecksum);
            sealedPage.SetBuffer(onDiskPage->GetAddress());
            R__ASSERT(onDiskPage && (onDiskPage->GetSize() == sealedPage.GetBufferSize()));

            auto taskFunc = [this, columnId, clusterId, firstInPage, sealedPage, element = allElements.back().get(),
                             &foundChecksumFailure,
                             indexOffset = clusterDescriptor.GetColumnRange(columnId).GetFirstElementIndex()]() {
               const ROOT::Internal::RPagePool::RKey keyPagePool{columnId, element->GetIdentifier().fInMemoryType};
               auto rv = UnsealPage(sealedPage, *element);
               if (!rv) {
                  foundChecksumFailure = true;
                  return;
               }
               auto newPage = rv.Unwrap();
               fCounters->fSzUnzip.Add(element->GetSize() * sealedPage.GetNElements());

               newPage.SetWindow(indexOffset + firstInPage,
                                 ROOT::Internal::RPage::RClusterInfo(clusterId, indexOffset));
               fPagePool.PreloadPage(std::move(newPage), keyPagePool);
            };

            fTaskScheduler->AddTask(taskFunc);

            firstInPage += pi.GetNElements();
            pageNo++;
         } // for all pages in column

         fCounters->fNPageUnsealed.Add(pageNo);
      } // for all in-memory types of the column
   } // for all columns in cluster

   fTaskScheduler->Wait();

   if (foundChecksumFailure) {
      throw RException(R__FAIL("page checksum verification failed, data corruption detected"));
   }
}

void ROOT::Internal::RPageSource::PrepareLoadCluster(
   const RCluster::RKey &clusterKey, ROnDiskPageMap &pageZeroMap,
   std::function<void(ROOT::DescriptorId_t, ROOT::NTupleSize_t, const ROOT::RClusterDescriptor::RPageInfo &)>
      perPageFunc)
{
   auto descriptorGuard = GetSharedDescriptorGuard();
   const auto &clusterDesc = descriptorGuard->GetClusterDescriptor(clusterKey.fClusterId);

   for (auto physicalColumnId : clusterKey.fPhysicalColumnSet) {
      if (clusterDesc.GetColumnRange(physicalColumnId).IsSuppressed())
         continue;

      const auto &pageRange = clusterDesc.GetPageRange(physicalColumnId);
      ROOT::NTupleSize_t pageNo = 0;
      for (const auto &pageInfo : pageRange.GetPageInfos()) {
         if (pageInfo.GetLocator().GetType() == RNTupleLocator::kTypePageZero) {
            pageZeroMap.Register(ROnDiskPage::Key{physicalColumnId, pageNo},
                                 ROnDiskPage(const_cast<void *>(ROOT::Internal::RPage::GetPageZeroBuffer()),
                                             pageInfo.GetLocator().GetNBytesOnStorage()));
         } else {
            perPageFunc(physicalColumnId, pageNo, pageInfo);
         }
         ++pageNo;
      }
   }
}

void ROOT::Internal::RPageSource::UpdateLastUsedCluster(ROOT::DescriptorId_t clusterId)
{
   if (fLastUsedCluster == clusterId)
      return;

   ROOT::NTupleSize_t firstEntryIndex =
      GetSharedDescriptorGuard()->GetClusterDescriptor(clusterId).GetFirstEntryIndex();
   auto itr = fPreloadedClusters.begin();
   while ((itr != fPreloadedClusters.end()) && (itr->first < firstEntryIndex)) {
      fPagePool.Evict(itr->second);
      itr = fPreloadedClusters.erase(itr);
   }
   std::size_t poolWindow = 0;
   while ((itr != fPreloadedClusters.end()) &&
          (poolWindow < 2 * ROOT::Internal::RNTupleReadOptionsManip::GetClusterBunchSize(fOptions))) {
      ++itr;
      ++poolWindow;
   }
   while (itr != fPreloadedClusters.end()) {
      fPagePool.Evict(itr->second);
      itr = fPreloadedClusters.erase(itr);
   }

   fLastUsedCluster = clusterId;
}

ROOT::Internal::RPageRef
ROOT::Internal::RPageSource::LoadPage(ColumnHandle_t columnHandle, ROOT::NTupleSize_t globalIndex)
{
   const auto columnId = columnHandle.fPhysicalId;
   const auto columnElementId = columnHandle.fColumn->GetElement()->GetIdentifier();
   auto cachedPageRef =
      fPagePool.GetPage(ROOT::Internal::RPagePool::RKey{columnId, columnElementId.fInMemoryType}, globalIndex);
   if (!cachedPageRef.Get().IsNull()) {
      UpdateLastUsedCluster(cachedPageRef.Get().GetClusterInfo().GetId());
      return cachedPageRef;
   }

   std::uint64_t idxInCluster;
   RClusterInfo clusterInfo;
   {
      auto descriptorGuard = GetSharedDescriptorGuard();
      clusterInfo.fClusterId = descriptorGuard->FindClusterId(columnId, globalIndex);

      if (clusterInfo.fClusterId == ROOT::kInvalidDescriptorId)
         throw RException(R__FAIL("entry with index " + std::to_string(globalIndex) + " out of bounds"));

      const auto &clusterDescriptor = descriptorGuard->GetClusterDescriptor(clusterInfo.fClusterId);
      const auto &columnRange = clusterDescriptor.GetColumnRange(columnId);
      if (columnRange.IsSuppressed())
         return ROOT::Internal::RPageRef();

      clusterInfo.fColumnOffset = columnRange.GetFirstElementIndex();
      R__ASSERT(clusterInfo.fColumnOffset <= globalIndex);
      idxInCluster = globalIndex - clusterInfo.fColumnOffset;
      clusterInfo.fPageInfo = clusterDescriptor.GetPageRange(columnId).Find(idxInCluster);
   }

   if (clusterInfo.fPageInfo.GetLocator().GetType() == RNTupleLocator::kTypeUnknown)
      throw RException(R__FAIL("tried to read a page with an unknown locator"));

   UpdateLastUsedCluster(clusterInfo.fClusterId);
   return LoadPageImpl(columnHandle, clusterInfo, idxInCluster);
}

ROOT::Internal::RPageRef
ROOT::Internal::RPageSource::LoadPage(ColumnHandle_t columnHandle, RNTupleLocalIndex localIndex)
{
   const auto clusterId = localIndex.GetClusterId();
   const auto idxInCluster = localIndex.GetIndexInCluster();
   const auto columnId = columnHandle.fPhysicalId;
   const auto columnElementId = columnHandle.fColumn->GetElement()->GetIdentifier();
   auto cachedPageRef =
      fPagePool.GetPage(ROOT::Internal::RPagePool::RKey{columnId, columnElementId.fInMemoryType}, localIndex);
   if (!cachedPageRef.Get().IsNull()) {
      UpdateLastUsedCluster(clusterId);
      return cachedPageRef;
   }

   if (clusterId == kInvalidDescriptorId)
      throw RException(R__FAIL("entry out of bounds"));

   RClusterInfo clusterInfo;
   {
      auto descriptorGuard = GetSharedDescriptorGuard();
      const auto &clusterDescriptor = descriptorGuard->GetClusterDescriptor(clusterId);
      const auto &columnRange = clusterDescriptor.GetColumnRange(columnId);
      if (columnRange.IsSuppressed())
         return ROOT::Internal::RPageRef();

      clusterInfo.fClusterId = clusterId;
      clusterInfo.fColumnOffset = columnRange.GetFirstElementIndex();
      clusterInfo.fPageInfo = clusterDescriptor.GetPageRange(columnId).Find(idxInCluster);
   }

   if (clusterInfo.fPageInfo.GetLocator().GetType() == RNTupleLocator::kTypeUnknown)
      throw RException(R__FAIL("tried to read a page with an unknown locator"));

   UpdateLastUsedCluster(clusterInfo.fClusterId);
   return LoadPageImpl(columnHandle, clusterInfo, idxInCluster);
}

void ROOT::Internal::RPageSource::EnableDefaultMetrics(const std::string &prefix)
{
   fMetrics = RNTupleMetrics(prefix);
   fCounters = std::make_unique<RCounters>(RCounters{
      *fMetrics.MakeCounter<RNTupleAtomicCounter *>("nReadV", "", "number of vector read requests"),
      *fMetrics.MakeCounter<RNTupleAtomicCounter *>("nRead", "", "number of byte ranges read"),
      *fMetrics.MakeCounter<RNTupleAtomicCounter *>("szReadPayload", "B", "volume read from storage (required)"),
      *fMetrics.MakeCounter<RNTupleAtomicCounter *>("szReadOverhead", "B", "volume read from storage (overhead)"),
      *fMetrics.MakeCounter<RNTupleAtomicCounter *>("szUnzip", "B", "volume after unzipping"),
      *fMetrics.MakeCounter<RNTupleAtomicCounter *>("nClusterLoaded", "",
                                                    "number of partial clusters preloaded from storage"),
      *fMetrics.MakeCounter<RNTupleAtomicCounter *>("nPageRead", "", "number of pages read from storage"),
      *fMetrics.MakeCounter<RNTupleAtomicCounter *>("nPageUnsealed", "", "number of pages unzipped and decoded"),
      *fMetrics.MakeCounter<RNTupleAtomicCounter *>("timeWallRead", "ns", "wall clock time spent reading"),
      *fMetrics.MakeCounter<RNTupleAtomicCounter *>("timeWallUnzip", "ns", "wall clock time spent decompressing"),
      *fMetrics.MakeCounter<RNTupleTickCounter<RNTupleAtomicCounter> *>("timeCpuRead", "ns", "CPU time spent reading"),
      *fMetrics.MakeCounter<RNTupleTickCounter<RNTupleAtomicCounter> *>("timeCpuUnzip", "ns",
                                                                        "CPU time spent decompressing"),
      *fMetrics.MakeCounter<RNTupleCalcPerf *>(
         "bwRead", "MB/s", "bandwidth compressed bytes read per second", fMetrics,
         [](const RNTupleMetrics &metrics) -> std::pair<bool, double> {
            if (const auto szReadPayload = metrics.GetLocalCounter("szReadPayload")) {
               if (const auto szReadOverhead = metrics.GetLocalCounter("szReadOverhead")) {
                  if (const auto timeWallRead = metrics.GetLocalCounter("timeWallRead")) {
                     if (auto walltime = timeWallRead->GetValueAsInt()) {
                        double payload = szReadPayload->GetValueAsInt();
                        double overhead = szReadOverhead->GetValueAsInt();
                        // unit: bytes / nanosecond = GB/s
                        return {true, (1000. * (payload + overhead) / walltime)};
                     }
                  }
               }
            }
            return {false, -1.};
         }),
      *fMetrics.MakeCounter<RNTupleCalcPerf *>(
         "bwReadUnzip", "MB/s", "bandwidth uncompressed bytes read per second", fMetrics,
         [](const RNTupleMetrics &metrics) -> std::pair<bool, double> {
            if (const auto szUnzip = metrics.GetLocalCounter("szUnzip")) {
               if (const auto timeWallRead = metrics.GetLocalCounter("timeWallRead")) {
                  if (auto walltime = timeWallRead->GetValueAsInt()) {
                     double unzip = szUnzip->GetValueAsInt();
                     // unit: bytes / nanosecond = GB/s
                     return {true, 1000. * unzip / walltime};
                  }
               }
            }
            return {false, -1.};
         }),
      *fMetrics.MakeCounter<RNTupleCalcPerf *>(
         "bwUnzip", "MB/s", "decompression bandwidth of uncompressed bytes per second", fMetrics,
         [](const RNTupleMetrics &metrics) -> std::pair<bool, double> {
            if (const auto szUnzip = metrics.GetLocalCounter("szUnzip")) {
               if (const auto timeWallUnzip = metrics.GetLocalCounter("timeWallUnzip")) {
                  if (auto walltime = timeWallUnzip->GetValueAsInt()) {
                     double unzip = szUnzip->GetValueAsInt();
                     // unit: bytes / nanosecond = GB/s
                     return {true, 1000. * unzip / walltime};
                  }
               }
            }
            return {false, -1.};
         }),
      *fMetrics.MakeCounter<RNTupleCalcPerf *>(
         "rtReadEfficiency", "", "ratio of payload over all bytes read", fMetrics,
         [](const RNTupleMetrics &metrics) -> std::pair<bool, double> {
            if (const auto szReadPayload = metrics.GetLocalCounter("szReadPayload")) {
               if (const auto szReadOverhead = metrics.GetLocalCounter("szReadOverhead")) {
                  if (auto payload = szReadPayload->GetValueAsInt()) {
                     // r/(r+o) = 1/((r+o)/r) = 1/(1 + o/r)
                     return {true, 1. / (1. + (1. * szReadOverhead->GetValueAsInt()) / payload)};
                  }
               }
            }
            return {false, -1.};
         }),
      *fMetrics.MakeCounter<RNTupleCalcPerf *>("rtCompression", "", "ratio of compressed bytes / uncompressed bytes",
                                               fMetrics, [](const RNTupleMetrics &metrics) -> std::pair<bool, double> {
                                                  if (const auto szReadPayload =
                                                         metrics.GetLocalCounter("szReadPayload")) {
                                                     if (const auto szUnzip = metrics.GetLocalCounter("szUnzip")) {
                                                        if (auto unzip = szUnzip->GetValueAsInt()) {
                                                           return {true, (1. * szReadPayload->GetValueAsInt()) / unzip};
                                                        }
                                                     }
                                                  }
                                                  return {false, -1.};
                                               })});
}

ROOT::RResult<ROOT::Internal::RPage>
ROOT::Internal::RPageSource::UnsealPage(const RSealedPage &sealedPage, const RColumnElementBase &element)
{
   return UnsealPage(sealedPage, element, *fPageAllocator);
}

ROOT::RResult<ROOT::Internal::RPage> ROOT::Internal::RPageSource::UnsealPage(const RSealedPage &sealedPage,
                                                                             const RColumnElementBase &element,
                                                                             ROOT::Internal::RPageAllocator &pageAlloc)
{
   // Unsealing a page zero is a no-op.  `RPageRange::ExtendToFitColumnRange()` guarantees that the page zero buffer is
   // large enough to hold `sealedPage.fNElements`
   if (sealedPage.GetBuffer() == ROOT::Internal::RPage::GetPageZeroBuffer()) {
      auto page = pageAlloc.NewPage(element.GetSize(), sealedPage.GetNElements());
      page.GrowUnchecked(sealedPage.GetNElements());
      memset(page.GetBuffer(), 0, page.GetNBytes());
      return page;
   }

   auto rv = sealedPage.VerifyChecksumIfEnabled();
   if (!rv)
      return R__FORWARD_ERROR(rv);

   const auto bytesPacked = element.GetPackedSize(sealedPage.GetNElements());
   auto page = pageAlloc.NewPage(element.GetPackedSize(), sealedPage.GetNElements());
   if (sealedPage.GetDataSize() != bytesPacked) {
      ROOT::Internal::RNTupleDecompressor::Unzip(sealedPage.GetBuffer(), sealedPage.GetDataSize(), bytesPacked,
                                                 page.GetBuffer());
   } else {
      // We cannot simply map the sealed page as we don't know its life time. Specialized page sources
      // may decide to implement to not use UnsealPage but to custom mapping / decompression code.
      // Note that usually pages are compressed.
      memcpy(page.GetBuffer(), sealedPage.GetBuffer(), bytesPacked);
   }

   if (!element.IsMappable()) {
      auto tmp = pageAlloc.NewPage(element.GetSize(), sealedPage.GetNElements());
      element.Unpack(tmp.GetBuffer(), page.GetBuffer(), sealedPage.GetNElements());
      page = std::move(tmp);
   }

   page.GrowUnchecked(sealedPage.GetNElements());
   return page;
}

void ROOT::Internal::RPageSource::RegisterStreamerInfos()
{
   if (fHasStreamerInfosRegistered)
      return;

   for (const auto &extraTypeInfo : fDescriptor.GetExtraTypeInfoIterable()) {
      if (extraTypeInfo.GetContentId() != EExtraTypeInfoIds::kStreamerInfo)
         continue;
      // We don't need the result, it's enough that during deserialization, BuildCheck() is called for every
      // streamer info record.
      RNTupleSerializer::DeserializeStreamerInfos(extraTypeInfo.GetContent()).Unwrap();
   }

   fHasStreamerInfosRegistered = true;
}

//------------------------------------------------------------------------------

bool ROOT::Internal::RWritePageMemoryManager::RColumnInfo::operator>(const RColumnInfo &other) const
{
   // Make the sort order unique by adding the physical on-disk column id as a secondary key
   if (fCurrentPageSize == other.fCurrentPageSize)
      return fColumn->GetOnDiskId() > other.fColumn->GetOnDiskId();
   return fCurrentPageSize > other.fCurrentPageSize;
}

bool ROOT::Internal::RWritePageMemoryManager::TryEvict(std::size_t targetAvailableSize, std::size_t pageSizeLimit)
{
   if (fMaxAllocatedBytes - fCurrentAllocatedBytes >= targetAvailableSize)
      return true;

   auto itr = fColumnsSortedByPageSize.begin();
   while (itr != fColumnsSortedByPageSize.end()) {
      if (itr->fCurrentPageSize <= pageSizeLimit)
         break;
      if (itr->fCurrentPageSize == itr->fInitialPageSize) {
         ++itr;
         continue;
      }

      // Flushing the current column will invalidate itr
      auto itrFlush = itr++;

      RColumnInfo next;
      if (itr != fColumnsSortedByPageSize.end())
         next = *itr;

      itrFlush->fColumn->Flush();
      if (fMaxAllocatedBytes - fCurrentAllocatedBytes >= targetAvailableSize)
         return true;

      if (next.fColumn == nullptr)
         return false;
      itr = fColumnsSortedByPageSize.find(next);
   };

   return false;
}

bool ROOT::Internal::RWritePageMemoryManager::TryUpdate(RColumn &column, std::size_t newWritePageSize)
{
   const RColumnInfo key{&column, column.GetWritePageCapacity(), 0};
   auto itr = fColumnsSortedByPageSize.find(key);
   if (itr == fColumnsSortedByPageSize.end()) {
      if (!TryEvict(newWritePageSize, 0))
         return false;
      fColumnsSortedByPageSize.insert({&column, newWritePageSize, newWritePageSize});
      fCurrentAllocatedBytes += newWritePageSize;
      return true;
   }

   RColumnInfo elem{*itr};
   assert(newWritePageSize >= elem.fInitialPageSize);

   if (newWritePageSize == elem.fCurrentPageSize)
      return true;

   fColumnsSortedByPageSize.erase(itr);

   if (newWritePageSize < elem.fCurrentPageSize) {
      // Page got smaller
      fCurrentAllocatedBytes -= elem.fCurrentPageSize - newWritePageSize;
      elem.fCurrentPageSize = newWritePageSize;
      fColumnsSortedByPageSize.insert(elem);
      return true;
   }

   // Page got larger, we may need to make space available
   const auto diffBytes = newWritePageSize - elem.fCurrentPageSize;
   if (!TryEvict(diffBytes, elem.fCurrentPageSize)) {
      // Don't change anything, let the calling column flush itself
      // TODO(jblomer): we may consider skipping the column in TryEvict and thus avoiding erase+insert
      fColumnsSortedByPageSize.insert(elem);
      return false;
   }
   fCurrentAllocatedBytes += diffBytes;
   elem.fCurrentPageSize = newWritePageSize;
   fColumnsSortedByPageSize.insert(elem);
   return true;
}

//------------------------------------------------------------------------------

ROOT::Internal::RPageSink::RPageSink(std::string_view name, const ROOT::RNTupleWriteOptions &options)
   : RPageStorage(name), fOptions(options.Clone()), fWritePageMemoryManager(options.GetPageBufferBudget())
{
   ROOT::Internal::EnsureValidNameForRNTuple(name, "RNTuple").ThrowOnError();
}

ROOT::Internal::RPageSink::~RPageSink() {}

ROOT::Internal::RPageStorage::RSealedPage ROOT::Internal::RPageSink::SealPage(const RSealPageConfig &config)
{
   assert(config.fPage);
   assert(config.fElement);
   assert(config.fBuffer);

   unsigned char *pageBuf = reinterpret_cast<unsigned char *>(config.fPage->GetBuffer());
   bool isAdoptedBuffer = true;
   auto nBytesPacked = config.fPage->GetNBytes();
   auto nBytesChecksum = config.fWriteChecksum * kNBytesPageChecksum;

   if (!config.fElement->IsMappable()) {
      nBytesPacked = config.fElement->GetPackedSize(config.fPage->GetNElements());
      pageBuf = new unsigned char[nBytesPacked];
      isAdoptedBuffer = false;
      config.fElement->Pack(pageBuf, config.fPage->GetBuffer(), config.fPage->GetNElements());
   }
   auto nBytesZipped = nBytesPacked;

   if ((config.fCompressionSettings != 0) || !config.fElement->IsMappable() || !config.fAllowAlias ||
       config.fWriteChecksum) {
      nBytesZipped =
         ROOT::Internal::RNTupleCompressor::Zip(pageBuf, nBytesPacked, config.fCompressionSettings, config.fBuffer);
      if (!isAdoptedBuffer)
         delete[] pageBuf;
      pageBuf = reinterpret_cast<unsigned char *>(config.fBuffer);
      isAdoptedBuffer = true;
   }

   R__ASSERT(isAdoptedBuffer);

   RSealedPage sealedPage{pageBuf, nBytesZipped + nBytesChecksum, config.fPage->GetNElements(), config.fWriteChecksum};
   sealedPage.ChecksumIfEnabled();

   return sealedPage;
}

ROOT::Internal::RPageStorage::RSealedPage
ROOT::Internal::RPageSink::SealPage(const ROOT::Internal::RPage &page, const RColumnElementBase &element)
{
   const auto nBytes = page.GetNBytes() + GetWriteOptions().GetEnablePageChecksums() * kNBytesPageChecksum;
   if (fSealPageBuffer.size() < nBytes)
      fSealPageBuffer.resize(nBytes);

   RSealPageConfig config;
   config.fPage = &page;
   config.fElement = &element;
   config.fCompressionSettings = GetWriteOptions().GetCompression();
   config.fWriteChecksum = GetWriteOptions().GetEnablePageChecksums();
   config.fAllowAlias = true;
   config.fBuffer = fSealPageBuffer.data();

   return SealPage(config);
}

void ROOT::Internal::RPageSink::CommitDataset()
{
   for (const auto &cb : fOnDatasetCommitCallbacks)
      cb(*this);
   CommitDatasetImpl();
}

ROOT::Internal::RPage ROOT::Internal::RPageSink::ReservePage(ColumnHandle_t columnHandle, std::size_t nElements)
{
   R__ASSERT(nElements > 0);
   const auto elementSize = columnHandle.fColumn->GetElement()->GetSize();
   const auto nBytes = elementSize * nElements;
   if (!fWritePageMemoryManager.TryUpdate(*columnHandle.fColumn, nBytes))
      return ROOT::Internal::RPage();
   return fPageAllocator->NewPage(elementSize, nElements);
}

//------------------------------------------------------------------------------

std::unique_ptr<ROOT::Internal::RPageSink>
ROOT::Internal::RPagePersistentSink::Create(std::string_view ntupleName, std::string_view location,
                                            const ROOT::RNTupleWriteOptions &options)
{
   if (ntupleName.empty()) {
      throw RException(R__FAIL("empty RNTuple name"));
   }
   if (location.empty()) {
      throw RException(R__FAIL("empty storage location"));
   }
   if (location.find("daos://") == 0) {
#ifdef R__ENABLE_DAOS
      return std::make_unique<ROOT::Experimental::Internal::RPageSinkDaos>(ntupleName, location, options);
#else
      throw RException(R__FAIL("This RNTuple build does not support DAOS."));
#endif
   }

   // Otherwise assume that the user wants us to create a file.
   return std::make_unique<ROOT::Internal::RPageSinkFile>(ntupleName, location, options);
}

ROOT::Internal::RPagePersistentSink::RPagePersistentSink(std::string_view name,
                                                         const ROOT::RNTupleWriteOptions &options)
   : RPageSink(name, options)
{
}

ROOT::Internal::RPagePersistentSink::~RPagePersistentSink() {}

ROOT::Internal::RPageStorage::ColumnHandle_t
ROOT::Internal::RPagePersistentSink::AddColumn(ROOT::DescriptorId_t fieldId, RColumn &column)
{
   auto columnId = fDescriptorBuilder.GetDescriptor().GetNPhysicalColumns();
   RColumnDescriptorBuilder columnBuilder;
   columnBuilder.LogicalColumnId(columnId)
      .PhysicalColumnId(columnId)
      .FieldId(fieldId)
      .BitsOnStorage(column.GetBitsOnStorage())
      .ValueRange(column.GetValueRange())
      .Type(column.GetType())
      .Index(column.GetIndex())
      .RepresentationIndex(column.GetRepresentationIndex())
      .FirstElementIndex(column.GetFirstElementIndex());
   // For late model extension, we assume that the primary column representation is the active one for the
   // deferred range. All other representations are suppressed.
   if (column.GetFirstElementIndex() > 0 && column.GetRepresentationIndex() > 0)
      columnBuilder.SetSuppressedDeferred();
   fDescriptorBuilder.AddColumn(columnBuilder.MakeDescriptor().Unwrap());
   return ColumnHandle_t{columnId, &column};
}

void ROOT::Internal::RPagePersistentSink::UpdateSchema(const ROOT::Internal::RNTupleModelChangeset &changeset,
                                                       ROOT::NTupleSize_t firstEntry)
{
   const auto &descriptor = fDescriptorBuilder.GetDescriptor();

   if (descriptor.GetNLogicalColumns() > descriptor.GetNPhysicalColumns()) {
      // If we already have alias columns, add an offset to the alias columns so that the new physical columns
      // of the changeset follow immediately the already existing physical columns
      auto getNColumns = [](const ROOT::RFieldBase &f) -> std::size_t {
         const auto &reps = f.GetColumnRepresentatives();
         if (reps.empty())
            return 0;
         return reps.size() * reps[0].size();
      };
      std::uint32_t nNewPhysicalColumns = 0;
      for (auto f : changeset.fAddedFields) {
         nNewPhysicalColumns += getNColumns(*f);
         for (const auto &descendant : *f)
            nNewPhysicalColumns += getNColumns(descendant);
      }
      fDescriptorBuilder.ShiftAliasColumns(nNewPhysicalColumns);
   }

   auto addField = [&](ROOT::RFieldBase &f) {
      auto fieldId = descriptor.GetNFields();
      fDescriptorBuilder.AddField(RFieldDescriptorBuilder::FromField(f).FieldId(fieldId).MakeDescriptor().Unwrap());
      fDescriptorBuilder.AddFieldLink(f.GetParent()->GetOnDiskId(), fieldId);
      f.SetOnDiskId(fieldId);
      ROOT::Internal::CallConnectPageSinkOnField(f, *this, firstEntry); // issues in turn calls to `AddColumn()`
   };
   auto addProjectedField = [&](ROOT::RFieldBase &f) {
      auto fieldId = descriptor.GetNFields();
      auto sourceFieldId =
         ROOT::Internal::GetProjectedFieldsOfModel(changeset.fModel).GetSourceField(&f)->GetOnDiskId();
      fDescriptorBuilder.AddField(RFieldDescriptorBuilder::FromField(f).FieldId(fieldId).MakeDescriptor().Unwrap());
      fDescriptorBuilder.AddFieldLink(f.GetParent()->GetOnDiskId(), fieldId);
      fDescriptorBuilder.AddFieldProjection(sourceFieldId, fieldId);
      f.SetOnDiskId(fieldId);
      for (const auto &source : descriptor.GetColumnIterable(sourceFieldId)) {
         auto targetId = descriptor.GetNLogicalColumns();
         RColumnDescriptorBuilder columnBuilder;
         columnBuilder.LogicalColumnId(targetId)
            .PhysicalColumnId(source.GetLogicalId())
            .FieldId(fieldId)
            .BitsOnStorage(source.GetBitsOnStorage())
            .ValueRange(source.GetValueRange())
            .Type(source.GetType())
            .Index(source.GetIndex())
            .RepresentationIndex(source.GetRepresentationIndex());
         fDescriptorBuilder.AddColumn(columnBuilder.MakeDescriptor().Unwrap());
      }
   };

   R__ASSERT(firstEntry >= fPrevClusterNEntries);
   const auto nColumnsBeforeUpdate = descriptor.GetNPhysicalColumns();
   for (auto f : changeset.fAddedFields) {
      addField(*f);
      for (auto &descendant : *f)
         addField(descendant);
   }
   for (auto f : changeset.fAddedProjectedFields) {
      addProjectedField(*f);
      for (auto &descendant : *f)
         addProjectedField(descendant);
   }

   const auto nColumns = descriptor.GetNPhysicalColumns();
   fOpenColumnRanges.reserve(fOpenColumnRanges.size() + (nColumns - nColumnsBeforeUpdate));
   fOpenPageRanges.reserve(fOpenPageRanges.size() + (nColumns - nColumnsBeforeUpdate));
   for (ROOT::DescriptorId_t i = nColumnsBeforeUpdate; i < nColumns; ++i) {
      ROOT::RClusterDescriptor::RColumnRange columnRange;
      columnRange.SetPhysicalColumnId(i);
      // We set the first element index in the current cluster to the first element that is part of a materialized page
      // (i.e., that is part of a page list). For columns created during late model extension, however, the column range
      // is fixed up as needed by `RClusterDescriptorBuilder::AddExtendedColumnRanges()` on read back.
      columnRange.SetFirstElementIndex(descriptor.GetColumnDescriptor(i).GetFirstElementIndex());
      columnRange.SetNElements(0);
      columnRange.SetCompressionSettings(GetWriteOptions().GetCompression());
      fOpenColumnRanges.emplace_back(columnRange);
      ROOT::RClusterDescriptor::RPageRange pageRange;
      pageRange.SetPhysicalColumnId(i);
      fOpenPageRanges.emplace_back(std::move(pageRange));
   }

   // Mapping of memory to on-disk column IDs usually happens during serialization of the ntuple header. If the
   // header was already serialized, this has to be done manually as it is required for page list serialization.
   if (fSerializationContext.GetHeaderSize() > 0)
      fSerializationContext.MapSchema(descriptor, /*forHeaderExtension=*/true);
}

void ROOT::Internal::RPagePersistentSink::UpdateExtraTypeInfo(const ROOT::RExtraTypeInfoDescriptor &extraTypeInfo)
{
   if (extraTypeInfo.GetContentId() != EExtraTypeInfoIds::kStreamerInfo)
      throw RException(R__FAIL("ROOT bug: unexpected type extra info in UpdateExtraTypeInfo()"));

   fStreamerInfos.merge(RNTupleSerializer::DeserializeStreamerInfos(extraTypeInfo.GetContent()).Unwrap());
}

void ROOT::Internal::RPagePersistentSink::InitImpl(ROOT::RNTupleModel &model)
{
   fDescriptorBuilder.SetNTuple(fNTupleName, model.GetDescription());
   const auto &descriptor = fDescriptorBuilder.GetDescriptor();

   auto &fieldZero = ROOT::Internal::GetFieldZeroOfModel(model);
   fDescriptorBuilder.AddField(RFieldDescriptorBuilder::FromField(fieldZero).FieldId(0).MakeDescriptor().Unwrap());
   fieldZero.SetOnDiskId(0);
   auto &projectedFields = ROOT::Internal::GetProjectedFieldsOfModel(model);
   projectedFields.GetFieldZero().SetOnDiskId(0);

   ROOT::Internal::RNTupleModelChangeset initialChangeset{model};
   initialChangeset.fAddedFields.reserve(fieldZero.GetMutableSubfields().size());
   for (auto f : fieldZero.GetMutableSubfields())
      initialChangeset.fAddedFields.emplace_back(f);
   initialChangeset.fAddedProjectedFields.reserve(projectedFields.GetFieldZero().GetMutableSubfields().size());
   for (auto f : projectedFields.GetFieldZero().GetMutableSubfields())
      initialChangeset.fAddedProjectedFields.emplace_back(f);
   UpdateSchema(initialChangeset, 0U);

   fSerializationContext = RNTupleSerializer::SerializeHeader(nullptr, descriptor).Unwrap();
   auto buffer = MakeUninitArray<unsigned char>(fSerializationContext.GetHeaderSize());
   fSerializationContext = RNTupleSerializer::SerializeHeader(buffer.get(), descriptor).Unwrap();
   InitImpl(buffer.get(), fSerializationContext.GetHeaderSize());

   fDescriptorBuilder.BeginHeaderExtension();
}

std::unique_ptr<ROOT::RNTupleModel>
ROOT::Internal::RPagePersistentSink::InitFromDescriptor(const ROOT::RNTupleDescriptor &srcDescriptor, bool copyClusters)
{
   // Create new descriptor
   fDescriptorBuilder.SetSchemaFromExisting(srcDescriptor);
   const auto &descriptor = fDescriptorBuilder.GetDescriptor();

   // Create column/page ranges
   const auto nColumns = descriptor.GetNPhysicalColumns();
   R__ASSERT(fOpenColumnRanges.empty() && fOpenPageRanges.empty());
   fOpenColumnRanges.reserve(nColumns);
   fOpenPageRanges.reserve(nColumns);
   for (ROOT::DescriptorId_t i = 0; i < nColumns; ++i) {
      const auto &column = descriptor.GetColumnDescriptor(i);
      ROOT::RClusterDescriptor::RColumnRange columnRange;
      columnRange.SetPhysicalColumnId(i);
      columnRange.SetFirstElementIndex(column.GetFirstElementIndex());
      columnRange.SetNElements(0);
      columnRange.SetCompressionSettings(GetWriteOptions().GetCompression());
      fOpenColumnRanges.emplace_back(columnRange);
      ROOT::RClusterDescriptor::RPageRange pageRange;
      pageRange.SetPhysicalColumnId(i);
      fOpenPageRanges.emplace_back(std::move(pageRange));
   }

   if (copyClusters) {
      // Clone and add all cluster descriptors
      auto clusterId = srcDescriptor.FindClusterId(0, 0);
      while (clusterId != ROOT::kInvalidDescriptorId) {
         auto &cluster = srcDescriptor.GetClusterDescriptor(clusterId);
         auto nEntries = cluster.GetNEntries();
         for (unsigned int i = 0; i < fOpenColumnRanges.size(); ++i) {
            R__ASSERT(fOpenColumnRanges[i].GetPhysicalColumnId() == i);
            if (!cluster.ContainsColumn(i)) // a cluster may not contain a column if that column is deferred
               break;
            const auto &columnRange = cluster.GetColumnRange(i);
            R__ASSERT(columnRange.GetPhysicalColumnId() == i);
            // TODO: properly handle suppressed columns (check MarkSuppressedColumnRange())
            fOpenColumnRanges[i].IncrementFirstElementIndex(columnRange.GetNElements());
         }
         fDescriptorBuilder.AddCluster(cluster.Clone());
         fPrevClusterNEntries += nEntries;

         clusterId = srcDescriptor.FindNextClusterId(clusterId);
      }
   }

   // Create model
   auto modelOpts = ROOT::RNTupleDescriptor::RCreateModelOptions();
   modelOpts.SetReconstructProjections(true);
   auto model = descriptor.CreateModel(modelOpts);
   if (!copyClusters) {
      auto &projectedFields = ROOT::Internal::GetProjectedFieldsOfModel(*model);
      projectedFields.GetFieldZero().SetOnDiskId(model->GetConstFieldZero().GetOnDiskId());
   }

   // Serialize header and init from it
   fSerializationContext = RNTupleSerializer::SerializeHeader(nullptr, descriptor).Unwrap();
   auto buffer = MakeUninitArray<unsigned char>(fSerializationContext.GetHeaderSize());
   fSerializationContext = RNTupleSerializer::SerializeHeader(buffer.get(), descriptor).Unwrap();
   InitImpl(buffer.get(), fSerializationContext.GetHeaderSize());

   fDescriptorBuilder.BeginHeaderExtension();

   // mark this sink as initialized
   fIsInitialized = true;

   return model;
}

void ROOT::Internal::RPagePersistentSink::CommitSuppressedColumn(ColumnHandle_t columnHandle)
{
   fOpenColumnRanges.at(columnHandle.fPhysicalId).SetIsSuppressed(true);
}

void ROOT::Internal::RPagePersistentSink::CommitPage(ColumnHandle_t columnHandle, const ROOT::Internal::RPage &page)
{
   fOpenColumnRanges.at(columnHandle.fPhysicalId).IncrementNElements(page.GetNElements());

   ROOT::RClusterDescriptor::RPageInfo pageInfo;
   pageInfo.SetNElements(page.GetNElements());
   pageInfo.SetLocator(CommitPageImpl(columnHandle, page));
   pageInfo.SetHasChecksum(GetWriteOptions().GetEnablePageChecksums());
   fOpenPageRanges.at(columnHandle.fPhysicalId).GetPageInfos().emplace_back(pageInfo);
}

void ROOT::Internal::RPagePersistentSink::CommitSealedPage(ROOT::DescriptorId_t physicalColumnId,
                                                           const RPageStorage::RSealedPage &sealedPage)
{
   fOpenColumnRanges.at(physicalColumnId).IncrementNElements(sealedPage.GetNElements());

   ROOT::RClusterDescriptor::RPageInfo pageInfo;
   pageInfo.SetNElements(sealedPage.GetNElements());
   pageInfo.SetLocator(CommitSealedPageImpl(physicalColumnId, sealedPage));
   pageInfo.SetHasChecksum(sealedPage.GetHasChecksum());
   fOpenPageRanges.at(physicalColumnId).GetPageInfos().emplace_back(pageInfo);
}

std::vector<ROOT::RNTupleLocator>
ROOT::Internal::RPagePersistentSink::CommitSealedPageVImpl(std::span<RPageStorage::RSealedPageGroup> ranges,
                                                           const std::vector<bool> &mask)
{
   std::vector<ROOT::RNTupleLocator> locators;
   locators.reserve(mask.size());
   std::size_t i = 0;
   for (auto &range : ranges) {
      for (auto sealedPageIt = range.fFirst; sealedPageIt != range.fLast; ++sealedPageIt) {
         if (mask[i++])
            locators.push_back(CommitSealedPageImpl(range.fPhysicalColumnId, *sealedPageIt));
      }
   }
   locators.shrink_to_fit();
   return locators;
}

void ROOT::Internal::RPagePersistentSink::CommitSealedPageV(std::span<RPageStorage::RSealedPageGroup> ranges)
{
   /// Used in the `originalPages` map
   struct RSealedPageLink {
      const RSealedPage *fSealedPage = nullptr; ///< Points to the first occurrence of a page with a specific checksum
      std::size_t fLocatorIdx = 0;              ///< The index in the locator vector returned by CommitSealedPageVImpl()
   };

   std::vector<bool> mask;
   // For every sealed page, stores the corresponding index in the locator vector returned by CommitSealedPageVImpl()
   std::vector<std::size_t> locatorIndexes;
   // Maps page checksums to the first sealed page with that checksum
   std::unordered_map<std::uint64_t, RSealedPageLink> originalPages;
   std::size_t iLocator = 0;
   for (auto &range : ranges) {
      const auto rangeSize = std::distance(range.fFirst, range.fLast);
      mask.reserve(mask.size() + rangeSize);
      locatorIndexes.reserve(locatorIndexes.size() + rangeSize);

      for (auto sealedPageIt = range.fFirst; sealedPageIt != range.fLast; ++sealedPageIt) {
         if (!fFeatures.fCanMergePages || !fOptions->GetEnableSamePageMerging()) {
            mask.emplace_back(true);
            locatorIndexes.emplace_back(iLocator++);
            continue;
         }
         // Same page merging requires page checksums - this is checked in the write options
         R__ASSERT(sealedPageIt->GetHasChecksum());

         const auto chk = sealedPageIt->GetChecksum().Unwrap();
         auto itr = originalPages.find(chk);
         if (itr == originalPages.end()) {
            originalPages.insert({chk, {&(*sealedPageIt), iLocator}});
            mask.emplace_back(true);
            locatorIndexes.emplace_back(iLocator++);
            continue;
         }

         const auto *p = itr->second.fSealedPage;
         if (sealedPageIt->GetDataSize() != p->GetDataSize() ||
             memcmp(sealedPageIt->GetBuffer(), p->GetBuffer(), p->GetDataSize())) {
            mask.emplace_back(true);
            locatorIndexes.emplace_back(iLocator++);
            continue;
         }

         mask.emplace_back(false);
         locatorIndexes.emplace_back(itr->second.fLocatorIdx);
      }

      mask.shrink_to_fit();
      locatorIndexes.shrink_to_fit();
   }

   auto locators = CommitSealedPageVImpl(ranges, mask);
   unsigned i = 0;

   for (auto &range : ranges) {
      for (auto sealedPageIt = range.fFirst; sealedPageIt != range.fLast; ++sealedPageIt) {
         fOpenColumnRanges.at(range.fPhysicalColumnId).IncrementNElements(sealedPageIt->GetNElements());

         ROOT::RClusterDescriptor::RPageInfo pageInfo;
         pageInfo.SetNElements(sealedPageIt->GetNElements());
         pageInfo.SetLocator(locators[locatorIndexes[i++]]);
         pageInfo.SetHasChecksum(sealedPageIt->GetHasChecksum());
         fOpenPageRanges.at(range.fPhysicalColumnId).GetPageInfos().emplace_back(pageInfo);
      }
   }
}

ROOT::Internal::RPageSink::RStagedCluster
ROOT::Internal::RPagePersistentSink::StageCluster(ROOT::NTupleSize_t nNewEntries)
{
   RStagedCluster stagedCluster;
   stagedCluster.fNBytesWritten = StageClusterImpl();
   stagedCluster.fNEntries = nNewEntries;

   for (unsigned int i = 0; i < fOpenColumnRanges.size(); ++i) {
      RStagedCluster::RColumnInfo columnInfo;
      columnInfo.fCompressionSettings = fOpenColumnRanges[i].GetCompressionSettings().value();
      if (fOpenColumnRanges[i].IsSuppressed()) {
         assert(fOpenPageRanges[i].GetPageInfos().empty());
         columnInfo.fPageRange.SetPhysicalColumnId(i);
         columnInfo.fIsSuppressed = true;
         // We reset suppressed columns to the state they would have if they were active (not suppressed).
         fOpenColumnRanges[i].SetNElements(0);
         fOpenColumnRanges[i].SetIsSuppressed(false);
      } else {
         std::swap(columnInfo.fPageRange, fOpenPageRanges[i]);
         fOpenPageRanges[i].SetPhysicalColumnId(i);

         columnInfo.fNElements = fOpenColumnRanges[i].GetNElements();
         fOpenColumnRanges[i].SetNElements(0);
      }
      stagedCluster.fColumnInfos.push_back(std::move(columnInfo));
   }

   return stagedCluster;
}

void ROOT::Internal::RPagePersistentSink::CommitStagedClusters(std::span<RStagedCluster> clusters)
{
   for (const auto &cluster : clusters) {
      RClusterDescriptorBuilder clusterBuilder;
      clusterBuilder.ClusterId(fDescriptorBuilder.GetDescriptor().GetNActiveClusters())
         .FirstEntryIndex(fPrevClusterNEntries)
         .NEntries(cluster.fNEntries);
      for (const auto &columnInfo : cluster.fColumnInfos) {
         const auto colId = columnInfo.fPageRange.GetPhysicalColumnId();
         if (columnInfo.fIsSuppressed) {
            assert(columnInfo.fPageRange.GetPageInfos().empty());
            clusterBuilder.MarkSuppressedColumnRange(colId);
         } else {
            clusterBuilder.CommitColumnRange(colId, fOpenColumnRanges[colId].GetFirstElementIndex(),
                                             columnInfo.fCompressionSettings, columnInfo.fPageRange);
            fOpenColumnRanges[colId].IncrementFirstElementIndex(columnInfo.fNElements);
         }
      }

      clusterBuilder.CommitSuppressedColumnRanges(fDescriptorBuilder.GetDescriptor()).ThrowOnError();
      for (const auto &columnInfo : cluster.fColumnInfos) {
         if (!columnInfo.fIsSuppressed)
            continue;
         const auto colId = columnInfo.fPageRange.GetPhysicalColumnId();
         // For suppressed columns, we need to reset the first element index to the first element of the next (upcoming)
         // cluster. This information has been determined for the committed cluster descriptor through
         // CommitSuppressedColumnRanges(), so we can use the information from the descriptor.
         const auto &columnRangeFromDesc = clusterBuilder.GetColumnRange(colId);
         fOpenColumnRanges[colId].SetFirstElementIndex(columnRangeFromDesc.GetFirstElementIndex() +
                                                       columnRangeFromDesc.GetNElements());
      }

      fDescriptorBuilder.AddCluster(clusterBuilder.MoveDescriptor().Unwrap());
      fPrevClusterNEntries += cluster.fNEntries;
   }
}

void ROOT::Internal::RPagePersistentSink::CommitClusterGroup()
{
   const auto &descriptor = fDescriptorBuilder.GetDescriptor();

   const auto nClusters = descriptor.GetNActiveClusters();
   std::vector<ROOT::DescriptorId_t> physClusterIDs;
   physClusterIDs.reserve(nClusters);
   for (auto i = fNextClusterInGroup; i < nClusters; ++i) {
      physClusterIDs.emplace_back(fSerializationContext.MapClusterId(i));
   }

   auto szPageList =
      RNTupleSerializer::SerializePageList(nullptr, descriptor, physClusterIDs, fSerializationContext).Unwrap();
   auto bufPageList = MakeUninitArray<unsigned char>(szPageList);
   RNTupleSerializer::SerializePageList(bufPageList.get(), descriptor, physClusterIDs, fSerializationContext);

   const auto clusterGroupId = descriptor.GetNClusterGroups();
   const auto locator = CommitClusterGroupImpl(bufPageList.get(), szPageList);
   RClusterGroupDescriptorBuilder cgBuilder;
   cgBuilder.ClusterGroupId(clusterGroupId).PageListLocator(locator).PageListLength(szPageList);
   if (fNextClusterInGroup == nClusters) {
      cgBuilder.MinEntry(0).EntrySpan(0).NClusters(0);
   } else {
      const auto &firstClusterDesc = descriptor.GetClusterDescriptor(fNextClusterInGroup);
      const auto &lastClusterDesc = descriptor.GetClusterDescriptor(nClusters - 1);
      cgBuilder.MinEntry(firstClusterDesc.GetFirstEntryIndex())
         .EntrySpan(lastClusterDesc.GetFirstEntryIndex() + lastClusterDesc.GetNEntries() -
                    firstClusterDesc.GetFirstEntryIndex())
         .NClusters(nClusters - fNextClusterInGroup);
   }
   std::vector<ROOT::DescriptorId_t> clusterIds;
   clusterIds.reserve(nClusters);
   for (auto i = fNextClusterInGroup; i < nClusters; ++i) {
      clusterIds.emplace_back(i);
   }
   cgBuilder.AddSortedClusters(clusterIds);
   fDescriptorBuilder.AddClusterGroup(cgBuilder.MoveDescriptor().Unwrap());
   fSerializationContext.MapClusterGroupId(clusterGroupId);

   fNextClusterInGroup = nClusters;
}

void ROOT::Internal::RPagePersistentSink::CommitDatasetImpl()
{
   if (!fStreamerInfos.empty()) {
      // De-duplicate extra type infos before writing. Usually we won't have them already in the descriptor, but
      // this may happen when we are writing back an already-existing RNTuple, e.g. when doing incremental merging.
      for (const auto &etDesc : fDescriptorBuilder.GetDescriptor().GetExtraTypeInfoIterable()) {
         if (etDesc.GetContentId() == EExtraTypeInfoIds::kStreamerInfo) {
            // The specification mandates that the type name for a kStreamerInfo should be empty and the type version
            // should be zero.
            R__ASSERT(etDesc.GetTypeName().empty());
            R__ASSERT(etDesc.GetTypeVersion() == 0);
            auto etInfo = RNTupleSerializer::DeserializeStreamerInfos(etDesc.GetContent()).Unwrap();
            fStreamerInfos.merge(etInfo);
         }
      }

      RExtraTypeInfoDescriptorBuilder extraInfoBuilder;
      extraInfoBuilder.ContentId(EExtraTypeInfoIds::kStreamerInfo)
         .Content(RNTupleSerializer::SerializeStreamerInfos(fStreamerInfos));
      fDescriptorBuilder.ReplaceExtraTypeInfo(extraInfoBuilder.MoveDescriptor().Unwrap());
   }

   const auto &descriptor = fDescriptorBuilder.GetDescriptor();

   auto szFooter = RNTupleSerializer::SerializeFooter(nullptr, descriptor, fSerializationContext).Unwrap();
   auto bufFooter = MakeUninitArray<unsigned char>(szFooter);
   RNTupleSerializer::SerializeFooter(bufFooter.get(), descriptor, fSerializationContext);

   CommitDatasetImpl(bufFooter.get(), szFooter);
}

void ROOT::Internal::RPagePersistentSink::EnableDefaultMetrics(const std::string &prefix)
{
   fMetrics = RNTupleMetrics(prefix);
   fCounters = std::make_unique<RCounters>(RCounters{
      *fMetrics.MakeCounter<RNTupleAtomicCounter *>("nPageCommitted", "", "number of pages committed to storage"),
      *fMetrics.MakeCounter<RNTupleAtomicCounter *>("szWritePayload", "B", "volume written for committed pages"),
      *fMetrics.MakeCounter<RNTupleAtomicCounter *>("szZip", "B", "volume before zipping"),
      *fMetrics.MakeCounter<RNTupleAtomicCounter *>("timeWallWrite", "ns", "wall clock time spent writing"),
      *fMetrics.MakeCounter<RNTupleAtomicCounter *>("timeWallZip", "ns", "wall clock time spent compressing"),
      *fMetrics.MakeCounter<RNTupleTickCounter<RNTupleAtomicCounter> *>("timeCpuWrite", "ns", "CPU time spent writing"),
      *fMetrics.MakeCounter<RNTupleTickCounter<RNTupleAtomicCounter> *>("timeCpuZip", "ns",
                                                                        "CPU time spent compressing")});
}
