import { BIT, create, createHistogram, isStr, clTH1, clTH2, clTH2F, kNoStats } from '../core.mjs';
import { ObjectPainter } from '../base/ObjectPainter.mjs';
import { TGraphPainter, clTGraphAsymmErrors } from '../hist2d/TGraphPainter.mjs';
import { TF1Painter } from './TF1Painter.mjs';
import { TH1Painter } from './TH1Painter.mjs';
import { TH2Painter } from './TH2Painter.mjs';
import { getTEfficiencyBoundaryFunc } from '../base/math.mjs';


const kIsBayesian = BIT(14),  // Bayesian statistics are used
      kPosteriorMode = BIT(15),  // Use posterior mean for best estimate (Bayesian statistics)
 //   kShortestInterval = BIT(16),  // Use shortest interval, not implemented - too complicated
      kUseBinPrior = BIT(17),  // Use a different prior for each bin
      kUseWeights = BIT(18),  // Use weights
      getBetaAlpha = (obj, bin) => (obj.fBeta_bin_params.length > bin) ? obj.fBeta_bin_params[bin].first : obj.fBeta_alpha,
      getBetaBeta = (obj, bin) => (obj.fBeta_bin_params.length > bin) ? obj.fBeta_bin_params[bin].second : obj.fBeta_beta;

/**
 * @summary Painter for TEfficiency object
 *
 * @private
 */

class TEfficiencyPainter extends ObjectPainter {

   /** @summary Calculate efficiency */
   getEfficiency(obj, bin) {
      const BetaMean = (a, b) => (a <= 0 || b <= 0) ? 0 : a / (a + b),
            BetaMode = (a, b) => {
         if (a <= 0 || b <= 0) return 0;
         if (a <= 1 || b <= 1) {
            if (a < b) return 0;
            if (a > b) return 1;
            if (a === b) return 0.5; // cannot do otherwise
         }
         return (a - 1.0) / (a + b -2.0);
      },
      total = obj.fTotalHistogram.fArray[bin], // should work for both 1-d and 2-d
      passed = obj.fPassedHistogram.fArray[bin]; // should work for both 1-d and 2-d

      if (obj.TestBit(kIsBayesian)) {
         // parameters for the beta prior distribution
         const alpha = obj.TestBit(kUseBinPrior) ? getBetaAlpha(obj, bin) : obj.fBeta_alpha,
               beta = obj.TestBit(kUseBinPrior) ? getBetaBeta(obj, bin) : obj.fBeta_beta;

         let aa, bb;
         if (obj.TestBit(kUseWeights)) {
            const tw = total, // fTotalHistogram->GetBinContent(bin);
                tw2 = obj.fTotalHistogram.fSumw2 ? obj.fTotalHistogram.fSumw2[bin] : Math.abs(total),
                pw = passed; // fPassedHistogram->GetBinContent(bin);

            if (tw2 <= 0) return pw/tw;

            // tw/tw2 re-normalize the weights
            const norm = tw/tw2;
            aa = pw * norm + alpha;
            bb = (tw - pw) * norm + beta;
         } else {
            aa = passed + alpha;
            bb = total - passed + beta;
         }

         return !obj.TestBit(kPosteriorMode) ? BetaMean(aa, bb) : BetaMode(aa, bb);
      }

      return total ? passed / total : 0;
   }

   /** @summary Calculate efficiency error low */
   getEfficiencyErrorLow(obj, bin, value) {
      const total = obj.fTotalHistogram.fArray[bin],
            passed = obj.fPassedHistogram.fArray[bin];
      let alpha = 0, beta = 0;
      if (obj.TestBit(kIsBayesian)) {
         alpha = obj.TestBit(kUseBinPrior) ? getBetaAlpha(obj, bin) : obj.fBeta_alpha;
         beta = obj.TestBit(kUseBinPrior) ? getBetaBeta(obj, bin) : obj.fBeta_beta;
      }

      return value - this.fBoundary(total, passed, obj.fConfLevel, false, alpha, beta);
   }

   /** @summary Calculate efficiency error low up */
   getEfficiencyErrorUp(obj, bin, value) {
      const total = obj.fTotalHistogram.fArray[bin],
            passed = obj.fPassedHistogram.fArray[bin];
      let alpha = 0, beta = 0;
      if (obj.TestBit(kIsBayesian)) {
         alpha = obj.TestBit(kUseBinPrior) ? getBetaAlpha(obj, bin) : obj.fBeta_alpha;
         beta = obj.TestBit(kUseBinPrior) ? getBetaBeta(obj, bin) : obj.fBeta_beta;
      }

      return this.fBoundary(total, passed, obj.fConfLevel, true, alpha, beta) - value;
   }

   /** @summary Copy drawing attributes */
   copyAttributes(obj, eff) {
      ['fLineColor', 'fLineStyle', 'fLineWidth', 'fFillColor', 'fFillStyle', 'fMarkerColor', 'fMarkerStyle', 'fMarkerSize'].forEach(name => { obj[name] = eff[name]; });
   }

   /** @summary Create graph for the drawing of 1-dim TEfficiency */
   createGraph(/* eff */) {
      const gr = create(clTGraphAsymmErrors);
      gr.fName = 'eff_graph';
      return gr;
   }

   /** @summary Create histogram for the drawing of 2-dim TEfficiency */
   createHisto(eff) {
      const nbinsx = eff.fTotalHistogram.fXaxis.fNbins,
            nbinsy = eff.fTotalHistogram.fYaxis.fNbins,
            hist = createHistogram(clTH2F, nbinsx, nbinsy);
      Object.assign(hist.fXaxis, eff.fTotalHistogram.fXaxis);
      Object.assign(hist.fYaxis, eff.fTotalHistogram.fYaxis);
      hist.fName = 'eff_histo';
      return hist;
   }

   /** @summary Fill graph with points from efficiency object */
   fillGraph(gr, opt) {
      const eff = this.getObject(),
            xaxis = eff.fTotalHistogram.fXaxis,
            npoints = xaxis.fNbins,
            plot0Bins = (opt.indexOf('e0') >= 0);

      for (let n = 0, j = 0; n < npoints; ++n) {
         if (!plot0Bins && eff.fTotalHistogram.getBinContent(n+1) === 0) continue;

         const value = this.getEfficiency(eff, n+1);

         gr.fX[j] = xaxis.GetBinCenter(n+1);
         gr.fY[j] = value;
         gr.fEXlow[j] = xaxis.GetBinCenter(n+1) - xaxis.GetBinLowEdge(n+1);
         gr.fEXhigh[j] = xaxis.GetBinLowEdge(n+2) - xaxis.GetBinCenter(n+1);
         gr.fEYlow[j] = this.getEfficiencyErrorLow(eff, n+1, value);
         gr.fEYhigh[j] = this.getEfficiencyErrorUp(eff, n+1, value);

         gr.fNpoints = ++j;
      }

      gr.fTitle = eff.fTitle;
      this.copyAttributes(gr, eff);
   }

   /** @summary Fill graph with points from efficiency object */
   fillHisto(hist) {
      const eff = this.getObject(),
            nbinsx = hist.fXaxis.fNbins,
            nbinsy = hist.fYaxis.fNbins;

      for (let i = 0; i < nbinsx+2; ++i) {
         for (let j = 0; j < nbinsy+2; ++j) {
            const bin = hist.getBin(i, j);
            hist.fArray[bin] = this.getEfficiency(eff, bin);
         }
      }

      hist.fTitle = eff.fTitle;
      hist.fBits |= kNoStats;
      this.copyAttributes(hist, eff);
   }

   /** @summary Draw function */
   drawFunction(indx) {
      const eff = this.getObject();

      if (!eff?.fFunctions || (indx >= eff.fFunctions.arr.length))
         return this;

      return TF1Painter.draw(this.getPadPainter(), eff.fFunctions.arr[indx], eff.fFunctions.opt[indx])
                        .then(funcp => {
                           funcp?.setSecondaryId(this, `func_${indx}`);
                           return this.drawFunction(indx + 1);
                        });
   }

   /** @summary Fill context menu */
   fillContextMenuItems(menu) {
      menu.addRedrawMenu(this);
   }

   /** @summary Fully redraw efficiency with new draw options */
   async redrawWith(opt, skip_cleanup) {
      if (!skip_cleanup)
         this.getPadPainter()?.removePrimitive(this, true);

      if (!opt || !isStr(opt)) opt = '';
      opt = opt.toLowerCase();

      let promise, draw_total = false;

      const eff = this.getObject(),
            dom = this.getDrawDom();

      if (opt[0] === 'b') {
         draw_total = true;
         promise = (this.ndim === 1 ? TH1Painter : TH2Painter).draw(dom, eff.fTotalHistogram, opt.slice(1));
      } else if (this.ndim === 1) {
         if (!opt) opt = 'ap';
         if ((opt.indexOf('same') < 0) && (opt.indexOf('a') < 0)) opt += 'a';
         if (opt.indexOf('p') < 0) opt += 'p';

         const gr = this.createGraph(eff);
         this.fillGraph(gr, opt);
         promise = TGraphPainter.draw(dom, gr, opt);
      } else {
         if (!opt) opt = 'col';
         const hist = this.createHisto(eff);
         this.fillHisto(hist, opt);
         promise = TH2Painter.draw(dom, hist, opt);
      }

      return promise.then(subp => {
         subp?.setSecondaryId(this, 'eff');
         this.addToPadPrimitives();
         return draw_total ? this : this.drawFunction(0);
      });
   }

   /** @summary Draw TEfficiency object */
   static async draw(dom, eff, opt) {
      if (!eff || !eff.fTotalHistogram)
         return null;

      const painter = new TEfficiencyPainter(dom, eff);

      if (eff.fTotalHistogram._typename.indexOf(clTH1) === 0)
         painter.ndim = 1;
      else if (eff.fTotalHistogram._typename.indexOf(clTH2) === 0)
         painter.ndim = 2;
      else
         return null;

      painter.fBoundary = getTEfficiencyBoundaryFunc(eff.fStatisticOption, eff.TestBit(kIsBayesian));

      return painter.redrawWith(opt, true);
   }

} // class TEfficiencyPainter

export { TEfficiencyPainter };
