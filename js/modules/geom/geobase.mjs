import { isObject, isFunc, isStr, BIT } from '../core.mjs';
import { THREE } from '../base/base3d.mjs';
import { createBufferGeometry, createNormal,
         Vertex as CsgVertex, Geometry as CsgGeometry, Polygon as CsgPolygon } from './csg.mjs';

const _cfg = {
   GradPerSegm: 6,       // grad per segment in cylinder/spherical symmetry shapes
   CompressComp: true    // use faces compression in composite shapes
};

/** @summary Returns or set geometry config values
 * @desc Supported 'GradPerSegm' and 'CompressComp'
 * @private */

function geoCfg(name, value) {
   if (value === undefined)
      return _cfg[name];

   _cfg[name] = value;
}

const kindGeo = 0,    // TGeoNode / TGeoShape
      kindEve = 1,    // TEveShape / TEveGeoShapeExtract
      kindShape = 2,  // special kind for single shape handling

      /** @summary TGeo-related bits
       * @private */
      geoBITS = {
         kVisOverride: BIT(0),  // volume's vis. attributes are overwritten
         kVisNone: BIT(1),  // the volume/node is invisible, as well as daughters
         kVisThis: BIT(2),  // this volume/node is visible
         kVisDaughters: BIT(3),  // all leaves are visible
         kVisOneLevel: BIT(4),  // first level daughters are visible (not used)
         kVisStreamed: BIT(5),  // true if attributes have been streamed
         kVisTouched: BIT(6),  // true if attributes are changed after closing geom
         kVisOnScreen: BIT(7),  // true if volume is visible on screen
         kVisContainers: BIT(12), // all containers visible
         kVisOnly: BIT(13), // just this visible
         kVisBranch: BIT(14), // only a given branch visible
         kVisRaytrace: BIT(15)  // raytracing flag
      },

      clTGeoBBox = 'TGeoBBox',
      clTGeoArb8 = 'TGeoArb8',
      clTGeoCone = 'TGeoCone',
      clTGeoConeSeg = 'TGeoConeSeg',
      clTGeoTube = 'TGeoTube',
      clTGeoTubeSeg = 'TGeoTubeSeg',
      clTGeoCtub = 'TGeoCtub',
      clTGeoTrd1 = 'TGeoTrd1',
      clTGeoTrd2 = 'TGeoTrd2',
      clTGeoPara = 'TGeoPara',
      clTGeoParaboloid = 'TGeoParaboloid',
      clTGeoPcon = 'TGeoPcon',
      clTGeoPgon = 'TGeoPgon',
      clTGeoShapeAssembly = 'TGeoShapeAssembly',
      clTGeoSphere = 'TGeoSphere',
      clTGeoTorus = 'TGeoTorus',
      clTGeoXtru = 'TGeoXtru',
      clTGeoTrap = 'TGeoTrap',
      clTGeoGtra = 'TGeoGtra',
      clTGeoEltu = 'TGeoEltu',
      clTGeoHype = 'TGeoHype',
      clTGeoCompositeShape = 'TGeoCompositeShape',
      clTGeoHalfSpace = 'TGeoHalfSpace',
      clTGeoScaledShape = 'TGeoScaledShape';

/** @summary Test fGeoAtt bits
  * @private */
function testGeoBit(volume, f) {
   const att = volume.fGeoAtt;
   return att === undefined ? false : Boolean(att & f);
}

/** @summary Set fGeoAtt bit
  * @private */
function setGeoBit(volume, f, value) {
   if (volume.fGeoAtt === undefined) return;
   volume.fGeoAtt = value ? (volume.fGeoAtt | f) : (volume.fGeoAtt & ~f);
}

/** @summary Toggle fGeoAttBit
  * @private */
function toggleGeoBit(volume, f) {
   if (volume.fGeoAtt !== undefined)
      volume.fGeoAtt ^= f & 0xffffff;
}

/** @summary Implementation of TGeoVolume::InvisibleAll
  * @private */
function setInvisibleAll(volume, flag) {
   if (flag === undefined) flag = true;

   setGeoBit(volume, geoBITS.kVisThis, !flag);
   // setGeoBit(this, geoBITS.kVisDaughters, !flag);

   if (volume.fNodes) {
      for (let n = 0; n < volume.fNodes.arr.length; ++n) {
         const sub = volume.fNodes.arr[n].fVolume;
         setGeoBit(sub, geoBITS.kVisThis, !flag);
         // setGeoBit(sub, geoBITS.kVisDaughters, !flag);
      }
   }
}

const _warn_msgs = {};

/** @summary method used to avoid duplication of warnings
 * @private */
function geoWarn(msg) {
   if (_warn_msgs[msg] !== undefined) return;
   _warn_msgs[msg] = true;
   console.warn(msg);
}

/** @summary Analyze TGeo node kind
 *  @desc  0 - TGeoNode
 *         1 - TEveGeoNode
 *        -1 - unsupported
 * @return detected node kind
 * @private */
function getNodeKind(obj) {
   if (!isObject(obj)) return -1;
   return ('fShape' in obj) && ('fTrans' in obj) ? kindEve : kindGeo;
}

/** @summary Returns number of shapes
  * @desc Used to count total shapes number in composites
  * @private */
function countNumShapes(shape) {
   if (!shape) return 0;
   if (shape._typename !== clTGeoCompositeShape) return 1;
   return countNumShapes(shape.fNode.fLeft) + countNumShapes(shape.fNode.fRight);
}


/** @summary Returns geo object name
  * @desc Can appends some special suffixes
  * @private */
function getObjectName(obj) {
   return obj?.fName ? (obj.fName + (obj.$geo_suffix || '')) : '';
}

/** @summary Check duplicates
  * @private */
function checkDuplicates(parent, chlds) {
   if (parent) {
      if (parent.$geo_checked) return;
      parent.$geo_checked = true;
   }

   const names = [], cnts = [];
   for (let k = 0; k < chlds.length; ++k) {
      const chld = chlds[k];
      if (!chld?.fName) continue;
      if (!chld.$geo_suffix) {
         const indx = names.indexOf(chld.fName);
         if (indx >= 0) {
            let cnt = cnts[indx] || 1;
            while (names.indexOf(chld.fName+'#'+cnt) >= 0) ++cnt;
            chld.$geo_suffix = '#' + cnt;
            cnts[indx] = cnt+1;
         }
      }
      names.push(getObjectName(chld));
   }
}


/** @summary Create normal to plane, defined with three points
  * @private */
function produceNormal(x1, y1, z1, x2, y2, z2, x3, y3, z3) {
   const pA = new THREE.Vector3(x1, y1, z1),
         pB = new THREE.Vector3(x2, y2, z2),
         pC = new THREE.Vector3(x3, y3, z3),
         cb = new THREE.Vector3(),
         ab = new THREE.Vector3();

   cb.subVectors(pC, pB);
   ab.subVectors(pA, pB);
   cb.cross(ab);

   return cb;
}

// ==========================================================================

/**
  * @summary Helper class for geometry creation
  *
  * @private
  */

class GeometryCreator {

   /** @summary Constructor
     * @param numfaces - number of faces */
   constructor(numfaces) {
      this.nfaces = numfaces;
      this.indx = 0;
      this.pos = new Float32Array(numfaces*9);
      this.norm = new Float32Array(numfaces*9);
   }

   /** @summary Add face with 3 vertices */
   addFace3(x1, y1, z1, x2, y2, z2, x3, y3, z3) {
      const indx = this.indx, pos = this.pos;
      pos[indx] = x1;
      pos[indx+1] = y1;
      pos[indx+2] = z1;
      pos[indx+3] = x2;
      pos[indx+4] = y2;
      pos[indx+5] = z2;
      pos[indx+6] = x3;
      pos[indx+7] = y3;
      pos[indx+8] = z3;
      this.last4 = false;
      this.indx = indx + 9;
   }

   /** @summary Start polygon */
   startPolygon() {}

   /** @summary Stop polygon */
   stopPolygon() {}

   /** @summary Add face with 4 vertices
     * @desc From four vertices one normally creates two faces (1,2,3) and (1,3,4)
     * if (reduce === 1), first face is reduced
     * if (reduce === 2), second face is reduced */
   addFace4(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, reduce) {
      let indx = this.indx;
      const pos = this.pos;

      if (reduce !== 1) {
         pos[indx] = x1;
         pos[indx+1] = y1;
         pos[indx+2] = z1;
         pos[indx+3] = x2;
         pos[indx+4] = y2;
         pos[indx+5] = z2;
         pos[indx+6] = x3;
         pos[indx+7] = y3;
         pos[indx+8] = z3;
         indx+=9;
      }

      if (reduce !== 2) {
         pos[indx] = x1;
         pos[indx+1] = y1;
         pos[indx+2] = z1;
         pos[indx+3] = x3;
         pos[indx+4] = y3;
         pos[indx+5] = z3;
         pos[indx+6] = x4;
         pos[indx+7] = y4;
         pos[indx+8] = z4;
         indx+=9;
      }

      this.last4 = (indx !== this.indx + 9);
      this.indx = indx;
   }

   /** @summary Specify normal for face with 4 vertices
     * @desc same as addFace4, assign normals for each individual vertex
     * reduce has same meaning and should be the same */
   setNormal4(nx1, ny1, nz1, nx2, ny2, nz2, nx3, ny3, nz3, nx4, ny4, nz4, reduce) {
      if (this.last4 && reduce)
         return console.error('missmatch between addFace4 and setNormal4 calls');

      let indx = this.indx - (this.last4 ? 18 : 9);
      const norm = this.norm;

      if (reduce !== 1) {
         norm[indx] = nx1;
         norm[indx+1] = ny1;
         norm[indx+2] = nz1;
         norm[indx+3] = nx2;
         norm[indx+4] = ny2;
         norm[indx+5] = nz2;
         norm[indx+6] = nx3;
         norm[indx+7] = ny3;
         norm[indx+8] = nz3;
         indx+=9;
      }

      if (reduce !== 2) {
         norm[indx] = nx1;
         norm[indx+1] = ny1;
         norm[indx+2] = nz1;
         norm[indx+3] = nx3;
         norm[indx+4] = ny3;
         norm[indx+5] = nz3;
         norm[indx+6] = nx4;
         norm[indx+7] = ny4;
         norm[indx+8] = nz4;
      }
   }

   /** @summary Recalculate Z with provided func */
   recalcZ(func) {
      const pos = this.pos,
            last = this.indx;
      let indx = last - (this.last4 ? 18 : 9);

      while (indx < last) {
         pos[indx+2] = func(pos[indx], pos[indx+1], pos[indx+2]);
         indx+=3;
      }
   }

   /** @summary Calculate normal */
   calcNormal() {
      if (!this.cb) {
         this.pA = new THREE.Vector3();
         this.pB = new THREE.Vector3();
         this.pC = new THREE.Vector3();
         this.cb = new THREE.Vector3();
         this.ab = new THREE.Vector3();
      }

      this.pA.fromArray(this.pos, this.indx - 9);
      this.pB.fromArray(this.pos, this.indx - 6);
      this.pC.fromArray(this.pos, this.indx - 3);

      this.cb.subVectors(this.pC, this.pB);
      this.ab.subVectors(this.pA, this.pB);
      this.cb.cross(this.ab);

      this.setNormal(this.cb.x, this.cb.y, this.cb.z);
   }

   /** @summary Set normal */
   setNormal(nx, ny, nz) {
      let indx = this.indx - 9;
      const norm = this.norm;

      norm[indx] = norm[indx+3] = norm[indx+6] = nx;
      norm[indx+1] = norm[indx+4] = norm[indx+7] = ny;
      norm[indx+2] = norm[indx+5] = norm[indx+8] = nz;

      if (this.last4) {
         indx -= 9;
         norm[indx] = norm[indx+3] = norm[indx+6] = nx;
         norm[indx+1] = norm[indx+4] = norm[indx+7] = ny;
         norm[indx+2] = norm[indx+5] = norm[indx+8] = nz;
      }
   }

   /** @summary Set normal
     * @desc special shortcut, when same normals can be applied for 1-2 point and 3-4 point */
   setNormal_12_34(nx12, ny12, nz12, nx34, ny34, nz34, reduce) {
      if (reduce === undefined) reduce = 0;

      let indx = this.indx - ((reduce > 0) ? 9 : 18);
      const norm = this.norm;

      if (reduce !== 1) {
         norm[indx] = nx12;
         norm[indx+1] = ny12;
         norm[indx+2] = nz12;
         norm[indx+3] = nx12;
         norm[indx+4] = ny12;
         norm[indx+5] = nz12;
         norm[indx+6] = nx34;
         norm[indx+7] = ny34;
         norm[indx+8] = nz34;
         indx += 9;
      }

      if (reduce !== 2) {
         norm[indx] = nx12;
         norm[indx+1] = ny12;
         norm[indx+2] = nz12;
         norm[indx+3] = nx34;
         norm[indx+4] = ny34;
         norm[indx+5] = nz34;
         norm[indx+6] = nx34;
         norm[indx+7] = ny34;
         norm[indx+8] = nz34;
      }
   }

   /** @summary Create geometry */
   create() {
      if (this.nfaces !== this.indx/9)
         console.error(`Mismatch with created ${this.nfaces} and filled ${this.indx/9} number of faces`);

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
      geometry.setAttribute('normal', new THREE.BufferAttribute(this.norm, 3));
      return geometry;
   }

}

// ================================================================================

/** @summary Helper class for CsgGeometry creation
  *
  * @private
  */

class PolygonsCreator {

   /** @summary constructor */
   constructor() {
      this.polygons = [];
   }

   /** @summary Start polygon */
   startPolygon(normal) {
      this.multi = 1;
      this.mnormal = normal;
   }

   /** @summary Stop polygon */
   stopPolygon() {
      if (!this.multi) return;
      this.multi = 0;
      console.error('Polygon should be already closed at this moment');
   }

   /** @summary Add face with 3 vertices */
   addFace3(x1, y1, z1, x2, y2, z2, x3, y3, z3) {
      this.addFace4(x1, y1, z1, x2, y2, z2, x3, y3, z3, x3, y3, z3, 2);
   }

   /** @summary Add face with 4 vertices
     * @desc From four vertices one normally creates two faces (1,2,3) and (1,3,4)
     * if (reduce === 1), first face is reduced
     * if (reduce === 2), second face is reduced */
   addFace4(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, reduce) {
      if (reduce === undefined) reduce = 0;

      this.v1 = new CsgVertex(x1, y1, z1, 0, 0, 0);
      this.v2 = (reduce === 1) ? null : new CsgVertex(x2, y2, z2, 0, 0, 0);
      this.v3 = new CsgVertex(x3, y3, z3, 0, 0, 0);
      this.v4 = (reduce === 2) ? null : new CsgVertex(x4, y4, z4, 0, 0, 0);

      this.reduce = reduce;

      if (this.multi) {
         if (reduce !== 2)
            console.error('polygon not supported for not-reduced faces');

         let polygon;

         if (this.multi++ === 1) {
            polygon = new CsgPolygon();

            polygon.vertices.push(this.mnormal ? this.v2 : this.v3);
            this.polygons.push(polygon);
         } else {
            polygon = this.polygons.at(-1);
            // check that last vertex equals to v2
            const last = this.mnormal ? polygon.vertices.at(-1) : polygon.vertices.at(0),
                  comp = this.mnormal ? this.v2 : this.v3;

            if (comp.diff(last) > 1e-12)
               console.error('vertex missmatch when building polygon');
         }

         const first = this.mnormal ? polygon.vertices[0] : polygon.vertices.at(-1),
               next = this.mnormal ? this.v3 : this.v2;

         if (next.diff(first) < 1e-12)
            this.multi = 0;
         else if (this.mnormal)
            polygon.vertices.push(this.v3);
          else
            polygon.vertices.unshift(this.v2);

         return;
      }

      const polygon = new CsgPolygon();

      switch (reduce) {
         case 0: polygon.vertices.push(this.v1, this.v2, this.v3, this.v4); break;
         case 1: polygon.vertices.push(this.v1, this.v3, this.v4); break;
         case 2: polygon.vertices.push(this.v1, this.v2, this.v3); break;
      }

      this.polygons.push(polygon);
   }

   /** @summary Specify normal for face with 4 vertices
     * @desc same as addFace4, assign normals for each individual vertex
     * reduce has same meaning and should be the same */
   setNormal4(nx1, ny1, nz1, nx2, ny2, nz2, nx3, ny3, nz3, nx4, ny4, nz4) {
      this.v1.setnormal(nx1, ny1, nz1);
      if (this.v2) this.v2.setnormal(nx2, ny2, nz2);
      this.v3.setnormal(nx3, ny3, nz3);
      if (this.v4) this.v4.setnormal(nx4, ny4, nz4);
   }

   /** @summary Set normal
     * @desc special shortcut, when same normals can be applied for 1-2 point and 3-4 point */
   setNormal_12_34(nx12, ny12, nz12, nx34, ny34, nz34) {
      this.v1.setnormal(nx12, ny12, nz12);
      if (this.v2) this.v2.setnormal(nx12, ny12, nz12);
      this.v3.setnormal(nx34, ny34, nz34);
      if (this.v4) this.v4.setnormal(nx34, ny34, nz34);
   }

   /** @summary Calculate normal */
   calcNormal() {
      if (!this.cb) {
         this.pA = new THREE.Vector3();
         this.pB = new THREE.Vector3();
         this.pC = new THREE.Vector3();
         this.cb = new THREE.Vector3();
         this.ab = new THREE.Vector3();
      }

      this.pA.set(this.v1.x, this.v1.y, this.v1.z);

      if (this.reduce !== 1) {
         this.pB.set(this.v2.x, this.v2.y, this.v2.z);
         this.pC.set(this.v3.x, this.v3.y, this.v3.z);
      } else {
         this.pB.set(this.v3.x, this.v3.y, this.v3.z);
         this.pC.set(this.v4.x, this.v4.y, this.v4.z);
      }

      this.cb.subVectors(this.pC, this.pB);
      this.ab.subVectors(this.pA, this.pB);
      this.cb.cross(this.ab);

      this.setNormal(this.cb.x, this.cb.y, this.cb.z);
   }

   /** @summary Set normal */
   setNormal(nx, ny, nz) {
      this.v1.setnormal(nx, ny, nz);
      if (this.v2) this.v2.setnormal(nx, ny, nz);
      this.v3.setnormal(nx, ny, nz);
      if (this.v4) this.v4.setnormal(nx, ny, nz);
   }

   /** @summary Recalculate Z with provided func */
   recalcZ(func) {
      this.v1.z = func(this.v1.x, this.v1.y, this.v1.z);
      if (this.v2) this.v2.z = func(this.v2.x, this.v2.y, this.v2.z);
      this.v3.z = func(this.v3.x, this.v3.y, this.v3.z);
      if (this.v4) this.v4.z = func(this.v4.x, this.v4.y, this.v4.z);
   }

   /** @summary Create geometry
     * @private */
   create() {
      return { polygons: this.polygons };
   }

}

// ================= all functions to create geometry ===================================

/** @summary Creates cube geometry
  * @private */
function createCubeBuffer(shape, faces_limit) {
   if (faces_limit < 0) return 12;

   const dx = shape.fDX, dy = shape.fDY, dz = shape.fDZ,
         creator = faces_limit ? new PolygonsCreator() : new GeometryCreator(12);

   creator.addFace4(dx, dy, dz, dx, -dy, dz, dx, -dy, -dz, dx, dy, -dz); creator.setNormal(1, 0, 0);

   creator.addFace4(-dx, dy, -dz, -dx, -dy, -dz, -dx, -dy, dz, -dx, dy, dz); creator.setNormal(-1, 0, 0);

   creator.addFace4(-dx, dy, -dz, -dx, dy, dz, dx, dy, dz, dx, dy, -dz); creator.setNormal(0, 1, 0);

   creator.addFace4(-dx, -dy, dz, -dx, -dy, -dz, dx, -dy, -dz, dx, -dy, dz); creator.setNormal(0, -1, 0);

   creator.addFace4(-dx, dy, dz, -dx, -dy, dz, dx, -dy, dz, dx, dy, dz); creator.setNormal(0, 0, 1);

   creator.addFace4(dx, dy, -dz, dx, -dy, -dz, -dx, -dy, -dz, -dx, dy, -dz); creator.setNormal(0, 0, -1);

   return creator.create();
}

/** @summary Creates 8 edges geometry
  * @private */
function create8edgesBuffer(v, faces_limit) {
   const indicies = [4, 7, 6, 5, 0, 3, 7, 4, 4, 5, 1, 0, 6, 2, 1, 5, 7, 3, 2, 6, 1, 2, 3, 0],
         creator = (faces_limit > 0) ? new PolygonsCreator() : new GeometryCreator(12);

   for (let n = 0; n < indicies.length; n += 4) {
      const i1 = indicies[n]*3,
            i2 = indicies[n+1]*3,
            i3 = indicies[n+2]*3,
            i4 = indicies[n+3]*3;
      creator.addFace4(v[i1], v[i1+1], v[i1+2], v[i2], v[i2+1], v[i2+2],
                       v[i3], v[i3+1], v[i3+2], v[i4], v[i4+1], v[i4+2]);
      if (n === 0)
         creator.setNormal(0, 0, 1);
      else if (n === 20)
         creator.setNormal(0, 0, -1);
      else
         creator.calcNormal();
   }

   return creator.create();
}

/** @summary Creates PARA geometry
  * @private */
function createParaBuffer(shape, faces_limit) {
   if (faces_limit < 0) return 12;

   const txy = shape.fTxy, txz = shape.fTxz, tyz = shape.fTyz, v = [
       -shape.fZ*txz-txy*shape.fY-shape.fX, -shape.fY-shape.fZ*tyz, -shape.fZ,
       -shape.fZ*txz+txy*shape.fY-shape.fX, shape.fY-shape.fZ*tyz, -shape.fZ,
       -shape.fZ*txz+txy*shape.fY+shape.fX, shape.fY-shape.fZ*tyz, -shape.fZ,
       -shape.fZ*txz-txy*shape.fY+shape.fX, -shape.fY-shape.fZ*tyz, -shape.fZ,
        shape.fZ*txz-txy*shape.fY-shape.fX, -shape.fY+shape.fZ*tyz, shape.fZ,
        shape.fZ*txz+txy*shape.fY-shape.fX, shape.fY+shape.fZ*tyz, shape.fZ,
        shape.fZ*txz+txy*shape.fY+shape.fX, shape.fY+shape.fZ*tyz, shape.fZ,
        shape.fZ*txz-txy*shape.fY+shape.fX, -shape.fY+shape.fZ*tyz, shape.fZ];

   return create8edgesBuffer(v, faces_limit);
}

/** @summary Creates trapezoid geometry
  * @private */
function createTrapezoidBuffer(shape, faces_limit) {
   if (faces_limit < 0) return 12;

   let y1, y2;
   if (shape._typename === clTGeoTrd1)
      y1 = y2 = shape.fDY;
   else {
      y1 = shape.fDy1; y2 = shape.fDy2;
   }

   const v = [
      -shape.fDx1, y1, -shape.fDZ,
       shape.fDx1, y1, -shape.fDZ,
       shape.fDx1, -y1, -shape.fDZ,
      -shape.fDx1, -y1, -shape.fDZ,
      -shape.fDx2, y2, shape.fDZ,
       shape.fDx2, y2, shape.fDZ,
       shape.fDx2, -y2, shape.fDZ,
      -shape.fDx2, -y2, shape.fDZ
   ];

   return create8edgesBuffer(v, faces_limit);
}


/** @summary Creates arb8 geometry
  * @private */
function createArb8Buffer(shape, faces_limit) {
   if (faces_limit < 0) return 12;

   const vertices = [
      shape.fXY[0][0], shape.fXY[0][1], -shape.fDZ,
      shape.fXY[1][0], shape.fXY[1][1], -shape.fDZ,
      shape.fXY[2][0], shape.fXY[2][1], -shape.fDZ,
      shape.fXY[3][0], shape.fXY[3][1], -shape.fDZ,
      shape.fXY[4][0], shape.fXY[4][1], shape.fDZ,
      shape.fXY[5][0], shape.fXY[5][1], shape.fDZ,
      shape.fXY[6][0], shape.fXY[6][1], shape.fDZ,
      shape.fXY[7][0], shape.fXY[7][1], shape.fDZ
   ],
    indicies = [
         4, 7, 6, 6, 5, 4, 3, 7, 4, 4, 0, 3,
         5, 1, 0, 0, 4, 5, 6, 2, 1, 1, 5, 6,
         7, 3, 2, 2, 6, 7, 1, 2, 3, 3, 0, 1];

   // detect same vertices on both Z-layers
   for (let side = 0; side < vertices.length; side += vertices.length/2) {
      for (let n1 = side; n1 < side + vertices.length/2 - 3; n1+=3) {
         for (let n2 = n1+3; n2 < side + vertices.length/2; n2+=3) {
             if ((vertices[n1] === vertices[n2]) &&
                (vertices[n1+1] === vertices[n2+1]) &&
                (vertices[n1+2] === vertices[n2+2])) {
                   for (let k=0; k<indicies.length; ++k)
                     if (indicies[k] === n2/3) indicies[k] = n1/3;
               }
         }
      }
   }

   const map = []; // list of existing faces (with all rotations)
   let numfaces = 0;

   for (let k = 0; k < indicies.length; k += 3) {
      const id1 = indicies[k]*100 + indicies[k+1]*10 + indicies[k+2],
            id2 = indicies[k+1]*100 + indicies[k+2]*10 + indicies[k],
            id3 = indicies[k+2]*100 + indicies[k]*10 + indicies[k+1];

      if ((indicies[k] === indicies[k+1]) || (indicies[k] === indicies[k+2]) || (indicies[k+1] === indicies[k+2]) ||
          (map.indexOf(id1) >= 0) || (map.indexOf(id2) >= 0) || (map.indexOf(id3) >= 0))
         indicies[k] = indicies[k+1] = indicies[k+2] = -1;
      else {
         map.push(id1, id2, id3);
         numfaces++;
      }
   }

   const creator = faces_limit ? new PolygonsCreator() : new GeometryCreator(numfaces);

   for (let n = 0; n < indicies.length; n += 6) {
      const i1 = indicies[n] * 3,
            i2 = indicies[n+1] * 3,
            i3 = indicies[n+2] * 3,
            i4 = indicies[n+3] * 3,
            i5 = indicies[n+4] * 3,
            i6 = indicies[n+5] * 3;
      let norm = null;

      if ((i1 >= 0) && (i4 >= 0) && faces_limit) {
         // try to identify two faces with same normal - very useful if one can create face4
         if (n === 0)
            norm = new THREE.Vector3(0, 0, 1);
         else if (n === 30)
            norm = new THREE.Vector3(0, 0, -1);
         else {
            const norm1 = produceNormal(vertices[i1], vertices[i1+1], vertices[i1+2],
                                      vertices[i2], vertices[i2+1], vertices[i2+2],
                                      vertices[i3], vertices[i3+1], vertices[i3+2]);

            norm1.normalize();

            const norm2 = produceNormal(vertices[i4], vertices[i4+1], vertices[i4+2],
                                      vertices[i5], vertices[i5+1], vertices[i5+2],
                                      vertices[i6], vertices[i6+1], vertices[i6+2]);

            norm2.normalize();

            if (norm1.distanceToSquared(norm2) < 1e-12) norm = norm1;
         }
      }

      if (norm !== null) {
         creator.addFace4(vertices[i1], vertices[i1+1], vertices[i1+2],
                          vertices[i2], vertices[i2+1], vertices[i2+2],
                          vertices[i3], vertices[i3+1], vertices[i3+2],
                          vertices[i5], vertices[i5+1], vertices[i5+2]);
         creator.setNormal(norm.x, norm.y, norm.z);
      } else {
         if (i1 >= 0) {
            creator.addFace3(vertices[i1], vertices[i1+1], vertices[i1+2],
                             vertices[i2], vertices[i2+1], vertices[i2+2],
                             vertices[i3], vertices[i3+1], vertices[i3+2]);
            creator.calcNormal();
         }
         if (i4 >= 0) {
            creator.addFace3(vertices[i4], vertices[i4+1], vertices[i4+2],
                             vertices[i5], vertices[i5+1], vertices[i5+2],
                             vertices[i6], vertices[i6+1], vertices[i6+2]);
            creator.calcNormal();
         }
      }
   }

   return creator.create();
}

/** @summary Creates sphere geometry
  * @private */
function createSphereBuffer(shape, faces_limit) {
   const radius = [shape.fRmax, shape.fRmin],
         phiStart = shape.fPhi1,
         phiLength = shape.fPhi2 - shape.fPhi1,
         thetaStart = shape.fTheta1,
         thetaLength = shape.fTheta2 - shape.fTheta1,
         noInside = (radius[1] <= 0);
   let widthSegments = shape.fNseg,
       heightSegments = shape.fNz;

   if (faces_limit > 0) {
      const fact = (noInside ? 2 : 4) * widthSegments * heightSegments / faces_limit;

      if (fact > 1.0) {
         widthSegments = Math.max(4, Math.floor(widthSegments/Math.sqrt(fact)));
         heightSegments = Math.max(4, Math.floor(heightSegments/Math.sqrt(fact)));
      }
   }

   let numoutside = widthSegments * heightSegments * 2,
       numtop = widthSegments * (noInside ? 1 : 2),
       numbottom = widthSegments * (noInside ? 1 : 2);
   const numcut = (phiLength === 360) ? 0 : heightSegments * (noInside ? 2 : 4),
         epsilon = 1e-10;

   if (faces_limit < 0) return numoutside * (noInside ? 1 : 2) + numtop + numbottom + numcut;

   const _sinp = new Float32Array(widthSegments+1),
       _cosp = new Float32Array(widthSegments+1),
       _sint = new Float32Array(heightSegments+1),
       _cost = new Float32Array(heightSegments+1);

   for (let n = 0; n <= heightSegments; ++n) {
      const theta = (thetaStart + thetaLength/heightSegments*n)*Math.PI/180;
      _sint[n] = Math.sin(theta);
      _cost[n] = Math.cos(theta);
   }

   for (let n = 0; n <= widthSegments; ++n) {
      const phi = (phiStart + phiLength/widthSegments*n)*Math.PI/180;
      _sinp[n] = Math.sin(phi);
      _cosp[n] = Math.cos(phi);
   }

   if (Math.abs(_sint[0]) <= epsilon) { numoutside -= widthSegments; numtop = 0; }
   if (Math.abs(_sint[heightSegments]) <= epsilon) { numoutside -= widthSegments; numbottom = 0; }

   const numfaces = numoutside * (noInside ? 1 : 2) + numtop + numbottom + numcut,
         creator = faces_limit ? new PolygonsCreator() : new GeometryCreator(numfaces);

   for (let side = 0; side < 2; ++side) {
      if ((side === 1) && noInside) break;

      const r = radius[side],
            s = (side === 0) ? 1 : -1,
            d1 = 1 - side, d2 = 1 - d1;

      // use direct algorithm for the sphere - here normals and position can be calculated directly
      for (let k = 0; k < heightSegments; ++k) {
         const k1 = k + d1, k2 = k + d2;
         let skip = 0;
         if (Math.abs(_sint[k1]) <= epsilon) skip = 1; else
         if (Math.abs(_sint[k2]) <= epsilon) skip = 2;

         for (let n = 0; n < widthSegments; ++n) {
            creator.addFace4(
                  r*_sint[k1]*_cosp[n], r*_sint[k1] *_sinp[n], r*_cost[k1],
                  r*_sint[k1]*_cosp[n+1], r*_sint[k1] *_sinp[n+1], r*_cost[k1],
                  r*_sint[k2]*_cosp[n+1], r*_sint[k2] *_sinp[n+1], r*_cost[k2],
                  r*_sint[k2]*_cosp[n], r*_sint[k2] *_sinp[n], r*_cost[k2],
                  skip);
            creator.setNormal4(
                  s*_sint[k1]*_cosp[n], s*_sint[k1] *_sinp[n], s*_cost[k1],
                  s*_sint[k1]*_cosp[n+1], s*_sint[k1] *_sinp[n+1], s*_cost[k1],
                  s*_sint[k2]*_cosp[n+1], s*_sint[k2] *_sinp[n+1], s*_cost[k2],
                  s*_sint[k2]*_cosp[n], s*_sint[k2] *_sinp[n], s*_cost[k2],
                  skip);
         }
      }
   }

   // top/bottom
   for (let side = 0; side <= heightSegments; side += heightSegments) {
      if (Math.abs(_sint[side]) >= epsilon) {
         const ss = _sint[side], cc = _cost[side],
               d1 = (side === 0) ? 0 : 1, d2 = 1 - d1;
         for (let n = 0; n < widthSegments; ++n) {
            creator.addFace4(
                  radius[1] * ss * _cosp[n+d1], radius[1] * ss * _sinp[n+d1], radius[1] * cc,
                  radius[0] * ss * _cosp[n+d1], radius[0] * ss * _sinp[n+d1], radius[0] * cc,
                  radius[0] * ss * _cosp[n+d2], radius[0] * ss * _sinp[n+d2], radius[0] * cc,
                  radius[1] * ss * _cosp[n+d2], radius[1] * ss * _sinp[n+d2], radius[1] * cc,
                  noInside ? 2 : 0);
            creator.calcNormal();
         }
      }
   }

   // cut left/right sides
   if (phiLength < 360) {
      for (let side = 0; side <= widthSegments; side += widthSegments) {
         const ss = _sinp[side], cc = _cosp[side],
               d1 = (side === 0) ? 1 : 0, d2 = 1 - d1;

         for (let k=0; k<heightSegments; ++k) {
            creator.addFace4(
                  radius[1] * _sint[k+d1] * cc, radius[1] * _sint[k+d1] * ss, radius[1] * _cost[k+d1],
                  radius[0] * _sint[k+d1] * cc, radius[0] * _sint[k+d1] * ss, radius[0] * _cost[k+d1],
                  radius[0] * _sint[k+d2] * cc, radius[0] * _sint[k+d2] * ss, radius[0] * _cost[k+d2],
                  radius[1] * _sint[k+d2] * cc, radius[1] * _sint[k+d2] * ss, radius[1] * _cost[k+d2],
                  noInside ? 2 : 0);
            creator.calcNormal();
         }
      }
   }

   return creator.create();
}

/** @summary Creates tube geometry
  * @private */
function createTubeBuffer(shape, faces_limit) {
   let outerR, innerR; // inner/outer tube radius
   if ((shape._typename === clTGeoCone) || (shape._typename === clTGeoConeSeg)) {
      outerR = [shape.fRmax2, shape.fRmax1];
      innerR = [shape.fRmin2, shape.fRmin1];
   } else {
      outerR = [shape.fRmax, shape.fRmax];
      innerR = [shape.fRmin, shape.fRmin];
   }

   const hasrmin = (innerR[0] > 0) || (innerR[1] > 0);
   let thetaStart = 0, thetaLength = 360;

   if ((shape._typename === clTGeoConeSeg) || (shape._typename === clTGeoTubeSeg) || (shape._typename === clTGeoCtub)) {
      thetaStart = shape.fPhi1;
      thetaLength = shape.fPhi2 - shape.fPhi1;
   }

   const radiusSegments = Math.max(4, Math.round(thetaLength / _cfg.GradPerSegm));

   // external surface
   let numfaces = radiusSegments * (((outerR[0] <= 0) || (outerR[1] <= 0)) ? 1 : 2);

   // internal surface
   if (hasrmin)
      numfaces += radiusSegments * (((innerR[0] <= 0) || (innerR[1] <= 0)) ? 1 : 2);

   // upper cap
   if (outerR[0] > 0) numfaces += radiusSegments * ((innerR[0] > 0) ? 2 : 1);
   // bottom cup
   if (outerR[1] > 0) numfaces += radiusSegments * ((innerR[1] > 0) ? 2 : 1);

   if (thetaLength < 360)
      numfaces += ((outerR[0] > innerR[0]) ? 2 : 0) + ((outerR[1] > innerR[1]) ? 2 : 0);

   if (faces_limit < 0) return numfaces;

   const phi0 = thetaStart*Math.PI/180,
         dphi = thetaLength/radiusSegments*Math.PI/180,
         _sin = new Float32Array(radiusSegments+1),
         _cos = new Float32Array(radiusSegments+1);

   for (let seg = 0; seg <= radiusSegments; ++seg) {
      _cos[seg] = Math.cos(phi0+seg*dphi);
      _sin[seg] = Math.sin(phi0+seg*dphi);
   }

   const creator = faces_limit ? new PolygonsCreator() : new GeometryCreator(numfaces),
   calcZ = (shape._typename !== clTGeoCtub)
      ? null
      : (x, y, z) => {
      const arr = (z < 0) ? shape.fNlow : shape.fNhigh;
      return ((z < 0) ? -shape.fDz : shape.fDz) - (x*arr[0] + y*arr[1]) / arr[2];
   };

   // create outer/inner tube
   for (let side = 0; side < 2; ++side) {
      if ((side === 1) && !hasrmin) break;

      const R = (side === 0) ? outerR : innerR, d1 = side, d2 = 1 - side;
      let nxy = 1, nz = 0;

      if (R[0] !== R[1]) {
         const angle = Math.atan2((R[1]-R[0]), 2*shape.fDZ);
         nxy = Math.cos(angle);
         nz = Math.sin(angle);
      }

      if (side === 1) { nxy *= -1; nz *= -1; }

      const reduce = (R[0] <= 0) ? 2 : ((R[1] <= 0) ? 1 : 0);

      for (let seg = 0; seg < radiusSegments; ++seg) {
         creator.addFace4(
               R[0] * _cos[seg+d1], R[0] * _sin[seg+d1], shape.fDZ,
               R[1] * _cos[seg+d1], R[1] * _sin[seg+d1], -shape.fDZ,
               R[1] * _cos[seg+d2], R[1] * _sin[seg+d2], -shape.fDZ,
               R[0] * _cos[seg+d2], R[0] * _sin[seg+d2], shape.fDZ,
               reduce);

         if (calcZ) creator.recalcZ(calcZ);

         creator.setNormal_12_34(nxy*_cos[seg+d1], nxy*_sin[seg+d1], nz,
                                 nxy*_cos[seg+d2], nxy*_sin[seg+d2], nz,
                                 reduce);
      }
   }

   // create upper/bottom part
   for (let side = 0; side < 2; ++side) {
      if (outerR[side] <= 0) continue;

      const d1 = side, d2 = 1- side,
          sign = (side === 0) ? 1 : -1,
          reduce = (innerR[side] <= 0) ? 2 : 0;
      if ((reduce === 2) && (thetaLength === 360) && !calcZ)
         creator.startPolygon(side === 0);
      for (let seg = 0; seg < radiusSegments; ++seg) {
         creator.addFace4(
               innerR[side] * _cos[seg+d1], innerR[side] * _sin[seg+d1], sign*shape.fDZ,
               outerR[side] * _cos[seg+d1], outerR[side] * _sin[seg+d1], sign*shape.fDZ,
               outerR[side] * _cos[seg+d2], outerR[side] * _sin[seg+d2], sign*shape.fDZ,
               innerR[side] * _cos[seg+d2], innerR[side] * _sin[seg+d2], sign*shape.fDZ,
               reduce);
         if (calcZ) {
            creator.recalcZ(calcZ);
            creator.calcNormal();
         } else
            creator.setNormal(0, 0, sign);
      }

      creator.stopPolygon();
   }

   // create cut surfaces
   if (thetaLength < 360) {
      creator.addFace4(innerR[1] * _cos[0], innerR[1] * _sin[0], -shape.fDZ,
                       outerR[1] * _cos[0], outerR[1] * _sin[0], -shape.fDZ,
                       outerR[0] * _cos[0], outerR[0] * _sin[0], shape.fDZ,
                       innerR[0] * _cos[0], innerR[0] * _sin[0], shape.fDZ,
                       (outerR[0] === innerR[0]) ? 2 : ((innerR[1] === outerR[1]) ? 1 : 0));
      if (calcZ) creator.recalcZ(calcZ);
      creator.calcNormal();

      creator.addFace4(innerR[0] * _cos[radiusSegments], innerR[0] * _sin[radiusSegments], shape.fDZ,
                       outerR[0] * _cos[radiusSegments], outerR[0] * _sin[radiusSegments], shape.fDZ,
                       outerR[1] * _cos[radiusSegments], outerR[1] * _sin[radiusSegments], -shape.fDZ,
                       innerR[1] * _cos[radiusSegments], innerR[1] * _sin[radiusSegments], -shape.fDZ,
                       (outerR[0] === innerR[0]) ? 1 : ((innerR[1] === outerR[1]) ? 2 : 0));

      if (calcZ) creator.recalcZ(calcZ);
      creator.calcNormal();
   }

   return creator.create();
}

/** @summary Creates eltu geometry
  * @private */
function createEltuBuffer(shape, faces_limit) {
   const radiusSegments = Math.max(4, Math.round(360 / _cfg.GradPerSegm));

   if (faces_limit < 0) return radiusSegments*4;

   // calculate all sin/cos tables in advance
   const x = new Float32Array(radiusSegments+1),
         y = new Float32Array(radiusSegments+1);
   for (let seg=0; seg<=radiusSegments; ++seg) {
      const phi = seg/radiusSegments*2*Math.PI;
      x[seg] = shape.fRmin*Math.cos(phi);
      y[seg] = shape.fRmax*Math.sin(phi);
   }

   const creator = faces_limit ? new PolygonsCreator() : new GeometryCreator(radiusSegments*4);
   let nx1, ny1, nx2 = 1, ny2 = 0;

   // create tube faces
   for (let seg = 0; seg < radiusSegments; ++seg) {
      creator.addFace4(x[seg], y[seg], shape.fDZ,
                       x[seg], y[seg], -shape.fDZ,
                       x[seg+1], y[seg+1], -shape.fDZ,
                       x[seg+1], y[seg+1], shape.fDZ);

      // calculate normals ourself
      nx1 = nx2; ny1 = ny2;
      nx2 = x[seg+1] * shape.fRmax / shape.fRmin;
      ny2 = y[seg+1] * shape.fRmin / shape.fRmax;
      const dist = Math.sqrt(nx2**2 + ny2**2);
      nx2 /= dist;
      ny2 /= dist;

      creator.setNormal_12_34(nx1, ny1, 0, nx2, ny2, 0);
   }

   // create top/bottom sides
   for (let side = 0; side < 2; ++side) {
      const sign = (side === 0) ? 1 : -1, d1 = side, d2 = 1 - side;
      for (let seg=0; seg<radiusSegments; ++seg) {
         creator.addFace3(0, 0, sign*shape.fDZ,
                          x[seg+d1], y[seg+d1], sign*shape.fDZ,
                          x[seg+d2], y[seg+d2], sign*shape.fDZ);
         creator.setNormal(0, 0, sign);
      }
   }

   return creator.create();
}

/** @summary Creates torus geometry
  * @private */
function createTorusBuffer(shape, faces_limit) {
   const radius = shape.fR;
   let radialSegments = Math.max(6, Math.round(360 / _cfg.GradPerSegm)),
       tubularSegments = Math.max(8, Math.round(shape.fDphi / _cfg.GradPerSegm)),
       numfaces = (shape.fRmin > 0 ? 4 : 2) * radialSegments * (tubularSegments + (shape.fDphi !== 360 ? 1 : 0));

   if (faces_limit < 0) return numfaces;

   if ((faces_limit > 0) && (numfaces > faces_limit)) {
      radialSegments = Math.floor(radialSegments/Math.sqrt(numfaces / faces_limit));
      tubularSegments = Math.floor(tubularSegments/Math.sqrt(numfaces / faces_limit));
      numfaces = (shape.fRmin > 0 ? 4 : 2) * radialSegments * (tubularSegments + (shape.fDphi !== 360 ? 1 : 0));
   }

   const _sinr = new Float32Array(radialSegments+1),
         _cosr = new Float32Array(radialSegments+1),
         _sint = new Float32Array(tubularSegments+1),
         _cost = new Float32Array(tubularSegments+1);

   for (let n = 0; n <= radialSegments; ++n) {
      _sinr[n] = Math.sin(n/radialSegments*2*Math.PI);
      _cosr[n] = Math.cos(n/radialSegments*2*Math.PI);
   }

   for (let t = 0; t <= tubularSegments; ++t) {
      const angle = (shape.fPhi1 + shape.fDphi*t/tubularSegments)/180*Math.PI;
      _sint[t] = Math.sin(angle);
      _cost[t] = Math.cos(angle);
   }

   const creator = faces_limit ? new PolygonsCreator() : new GeometryCreator(numfaces),
         // use vectors for normals calculation
         p1 = new THREE.Vector3(), p2 = new THREE.Vector3(), p3 = new THREE.Vector3(), p4 = new THREE.Vector3(),
         n1 = new THREE.Vector3(), n2 = new THREE.Vector3(), n3 = new THREE.Vector3(), n4 = new THREE.Vector3(),
         center1 = new THREE.Vector3(), center2 = new THREE.Vector3();

   for (let side = 0; side < 2; ++side) {
      if ((side > 0) && (shape.fRmin <= 0)) break;
      const tube = (side > 0) ? shape.fRmin : shape.fRmax,
            d1 = 1 - side, d2 = 1 - d1, ns = side > 0 ? -1 : 1;

      for (let t = 0; t < tubularSegments; ++t) {
         const t1 = t + d1, t2 = t + d2;
         center1.x = radius * _cost[t1]; center1.y = radius * _sint[t1];
         center2.x = radius * _cost[t2]; center2.y = radius * _sint[t2];

         for (let n = 0; n < radialSegments; ++n) {
            p1.x = (radius + tube * _cosr[n]) * _cost[t1]; p1.y = (radius + tube * _cosr[n]) * _sint[t1]; p1.z = tube*_sinr[n];
            p2.x = (radius + tube * _cosr[n+1]) * _cost[t1]; p2.y = (radius + tube * _cosr[n+1]) * _sint[t1]; p2.z = tube*_sinr[n+1];
            p3.x = (radius + tube * _cosr[n+1]) * _cost[t2]; p3.y = (radius + tube * _cosr[n+1]) * _sint[t2]; p3.z = tube*_sinr[n+1];
            p4.x = (radius + tube * _cosr[n]) * _cost[t2]; p4.y = (radius + tube * _cosr[n]) * _sint[t2]; p4.z = tube*_sinr[n];

            creator.addFace4(p1.x, p1.y, p1.z,
                             p2.x, p2.y, p2.z,
                             p3.x, p3.y, p3.z,
                             p4.x, p4.y, p4.z);

            n1.subVectors(p1, center1).normalize();
            n2.subVectors(p2, center1).normalize();
            n3.subVectors(p3, center2).normalize();
            n4.subVectors(p4, center2).normalize();

            creator.setNormal4(ns*n1.x, ns*n1.y, ns*n1.z,
                               ns*n2.x, ns*n2.y, ns*n2.z,
                               ns*n3.x, ns*n3.y, ns*n3.z,
                               ns*n4.x, ns*n4.y, ns*n4.z);
         }
      }
   }

   if (shape.fDphi !== 360) {
      for (let t = 0; t <= tubularSegments; t += tubularSegments) {
         const tube1 = shape.fRmax, tube2 = shape.fRmin,
               d1 = t > 0 ? 0 : 1, d2 = 1 - d1,
               skip = shape.fRmin > 0 ? 0 : 1,
               nsign = t > 0 ? 1 : -1;
         for (let n = 0; n < radialSegments; ++n) {
            creator.addFace4((radius + tube1 * _cosr[n+d1]) * _cost[t], (radius + tube1 * _cosr[n+d1]) * _sint[t], tube1*_sinr[n+d1],
                             (radius + tube2 * _cosr[n+d1]) * _cost[t], (radius + tube2 * _cosr[n+d1]) * _sint[t], tube2*_sinr[n+d1],
                             (radius + tube2 * _cosr[n+d2]) * _cost[t], (radius + tube2 * _cosr[n+d2]) * _sint[t], tube2*_sinr[n+d2],
                             (radius + tube1 * _cosr[n+d2]) * _cost[t], (radius + tube1 * _cosr[n+d2]) * _sint[t], tube1*_sinr[n+d2], skip);
            creator.setNormal(-nsign * _sint[t], nsign * _cost[t], 0);
         }
      }
   }

   return creator.create();
}


/** @summary Creates polygon geometry
  * @private */
function createPolygonBuffer(shape, faces_limit) {
   const thetaStart = shape.fPhi1,
         thetaLength = shape.fDphi;
   let radiusSegments, factor;

   if (shape._typename === clTGeoPgon) {
      radiusSegments = shape.fNedges;
      factor = 1.0 / Math.cos(Math.PI/180 * thetaLength / radiusSegments / 2);
   } else {
      radiusSegments = Math.max(5, Math.round(thetaLength / _cfg.GradPerSegm));
      factor = 1;
   }

   const usage = new Int16Array(2*shape.fNz);
   let numusedlayers = 0, hasrmin = false;

   for (let layer = 0; layer < shape.fNz; ++layer)
      hasrmin = hasrmin || (shape.fRmin[layer] > 0);

   // return very rough estimation, number of faces may be much less
   if (faces_limit < 0)
      return (hasrmin ? 4 : 2) * radiusSegments * (shape.fNz-1);

   // coordinate of point on cut edge (x,z)
   const pnts = (thetaLength === 360) ? null : [];

   // first analyze levels - if we need to create all of them
   for (let side = 0; side < 2; ++side) {
      const rside = (side === 0) ? 'fRmax' : 'fRmin';

      for (let layer=0; layer < shape.fNz; ++layer) {
         // first create points for the layer
         const layerz = shape.fZ[layer], rad = shape[rside][layer];

         usage[layer*2+side] = 0;

         if ((layer > 0) && (layer < shape.fNz-1)) {
            if (((shape.fZ[layer-1] === layerz) && (shape[rside][layer-1] === rad)) ||
                ((shape[rside][layer+1] === rad) && (shape[rside][layer-1] === rad))) {
               // same Z and R as before - ignore
               // or same R before and after

               continue;
            }
         }

         if ((layer > 0) && ((side === 0) || hasrmin)) {
            usage[layer*2+side] = 1;
            numusedlayers++;
         }

         if (pnts !== null) {
            if (side === 0)
               pnts.push(new THREE.Vector2(factor*rad, layerz));
             else if (rad < shape.fRmax[layer])
               pnts.unshift(new THREE.Vector2(factor*rad, layerz));
         }
      }
   }

   let numfaces = numusedlayers*radiusSegments*2;
   if (shape.fRmin[0] !== shape.fRmax[0])
      numfaces += radiusSegments * (hasrmin ? 2 : 1);
   if (shape.fRmin[shape.fNz-1] !== shape.fRmax[shape.fNz-1])
      numfaces += radiusSegments * (hasrmin ? 2 : 1);

   let cut_faces = null;

   if (pnts !== null) {
      if (pnts.length === shape.fNz * 2) {
         // special case - all layers are there, create faces ourself
         cut_faces = [];
         for (let layer = shape.fNz-1; layer > 0; --layer) {
            if (shape.fZ[layer] === shape.fZ[layer-1]) continue;
            const right = 2*shape.fNz - 1 - layer;
            cut_faces.push([right, layer - 1, layer]);
            cut_faces.push([right, right + 1, layer-1]);
         }
      } else {
         // let three.js calculate our faces
         cut_faces = THREE.ShapeUtils.triangulateShape(pnts, []);
      }
      numfaces += cut_faces.length*2;
   }

   const phi0 = thetaStart*Math.PI/180,
         dphi = thetaLength/radiusSegments*Math.PI/180,
         // calculate all sin/cos tables in advance
         _sin = new Float32Array(radiusSegments+1),
         _cos = new Float32Array(radiusSegments+1);
   for (let seg = 0; seg <= radiusSegments; ++seg) {
      _cos[seg] = Math.cos(phi0+seg*dphi);
      _sin[seg] = Math.sin(phi0+seg*dphi);
   }

   const creator = faces_limit ? new PolygonsCreator() : new GeometryCreator(numfaces);

   // add sides
   for (let side = 0; side < 2; ++side) {
      const rside = (side === 0) ? 'fRmax' : 'fRmin',
            d1 = 1 - side, d2 = side;
      let z1 = shape.fZ[0], r1 = factor*shape[rside][0];

      for (let layer = 0; layer < shape.fNz; ++layer) {
         if (usage[layer*2+side] === 0) continue;

         const z2 = shape.fZ[layer], r2 = factor*shape[rside][layer];
         let nxy = 1, nz = 0;

         if ((r2 !== r1)) {
            const angle = Math.atan2((r2-r1), (z2-z1));
            nxy = Math.cos(angle);
            nz = Math.sin(angle);
         }

         if (side > 0) { nxy*=-1; nz*=-1; }

         for (let seg = 0; seg < radiusSegments; ++seg) {
            creator.addFace4(r1 * _cos[seg+d1], r1 * _sin[seg+d1], z1,
                             r2 * _cos[seg+d1], r2 * _sin[seg+d1], z2,
                             r2 * _cos[seg+d2], r2 * _sin[seg+d2], z2,
                             r1 * _cos[seg+d2], r1 * _sin[seg+d2], z1);
            creator.setNormal_12_34(nxy*_cos[seg+d1], nxy*_sin[seg+d1], nz, nxy*_cos[seg+d2], nxy*_sin[seg+d2], nz);
         }

         z1 = z2; r1 = r2;
      }
   }

   // add top/bottom
   for (let layer = 0; layer < shape.fNz; layer += (shape.fNz-1)) {
      const rmin = factor*shape.fRmin[layer], rmax = factor*shape.fRmax[layer];

      if (rmin === rmax) continue;

      const layerz = shape.fZ[layer],
            d1 = (layer === 0) ? 1 : 0, d2 = 1 - d1,
            normalz = (layer === 0) ? -1: 1;

      if (!hasrmin && !cut_faces)
         creator.startPolygon(layer > 0);

      for (let seg = 0; seg < radiusSegments; ++seg) {
         creator.addFace4(rmin * _cos[seg+d1], rmin * _sin[seg+d1], layerz,
                          rmax * _cos[seg+d1], rmax * _sin[seg+d1], layerz,
                          rmax * _cos[seg+d2], rmax * _sin[seg+d2], layerz,
                          rmin * _cos[seg+d2], rmin * _sin[seg+d2], layerz,
                          hasrmin ? 0 : 2);
         creator.setNormal(0, 0, normalz);
      }

      creator.stopPolygon();
   }

   if (cut_faces) {
      for (let seg = 0; seg <= radiusSegments; seg += radiusSegments) {
         const d1 = (seg === 0) ? 1 : 2, d2 = 3 - d1;
         for (let n=0; n<cut_faces.length; ++n) {
            const a = pnts[cut_faces[n][0]],
                b = pnts[cut_faces[n][d1]],
                c = pnts[cut_faces[n][d2]];

            creator.addFace3(a.x * _cos[seg], a.x * _sin[seg], a.y,
                             b.x * _cos[seg], b.x * _sin[seg], b.y,
                             c.x * _cos[seg], c.x * _sin[seg], c.y);

            creator.calcNormal();
         }
      }
   }

   return creator.create();
}

/** @summary Creates xtru geometry
  * @private */
function createXtruBuffer(shape, faces_limit) {
   let nfaces = (shape.fNz-1) * shape.fNvert * 2;

   if (faces_limit < 0)
      return nfaces + shape.fNvert*3;

   // create points
   const pnts = [];
   for (let vert = 0; vert < shape.fNvert; ++vert)
      pnts.push(new THREE.Vector2(shape.fX[vert], shape.fY[vert]));

   let faces = THREE.ShapeUtils.triangulateShape(pnts, []);
   if (faces.length < pnts.length - 2) {
      geoWarn(`Problem with XTRU shape ${shape.fName} with ${pnts.length} vertices`);
      faces = [];
   } else
      nfaces += faces.length * 2;

   const creator = faces_limit ? new PolygonsCreator() : new GeometryCreator(nfaces);

   for (let layer = 0; layer < shape.fNz-1; ++layer) {
      const z1 = shape.fZ[layer], scale1 = shape.fScale[layer],
            z2 = shape.fZ[layer+1], scale2 = shape.fScale[layer+1],
            x01 = shape.fX0[layer], x02 = shape.fX0[layer+1],
            y01 = shape.fY0[layer], y02 = shape.fY0[layer+1];

      for (let vert1 = 0; vert1 < shape.fNvert; ++vert1) {
         const vert2 = (vert1+1) % shape.fNvert;
         creator.addFace4(scale1 * shape.fX[vert1] + x01, scale1 * shape.fY[vert1] + y01, z1,
                          scale2 * shape.fX[vert1] + x02, scale2 * shape.fY[vert1] + y02, z2,
                          scale2 * shape.fX[vert2] + x02, scale2 * shape.fY[vert2] + y02, z2,
                          scale1 * shape.fX[vert2] + x01, scale1 * shape.fY[vert2] + y01, z1);
         creator.calcNormal();
      }
   }

   for (let layer = 0; layer <= shape.fNz-1; layer += (shape.fNz-1)) {
      const z = shape.fZ[layer], scale = shape.fScale[layer],
            x0 = shape.fX0[layer], y0 = shape.fY0[layer];

      for (let n = 0; n < faces.length; ++n) {
         const face = faces[n],
               pnt1 = pnts[face[0]],
               pnt2 = pnts[face[layer === 0 ? 2 : 1]],
               pnt3 = pnts[face[layer === 0 ? 1 : 2]];

         creator.addFace3(scale * pnt1.x + x0, scale * pnt1.y + y0, z,
                          scale * pnt2.x + x0, scale * pnt2.y + y0, z,
                          scale * pnt3.x + x0, scale * pnt3.y + y0, z);
         creator.setNormal(0, 0, layer === 0 ? -1 : 1);
      }
   }

   return creator.create();
}

/** @summary Creates para geometry
  * @private */
function createParaboloidBuffer(shape, faces_limit) {
   let radiusSegments = Math.max(4, Math.round(360 / _cfg.GradPerSegm)),
       heightSegments = 30;

   if (faces_limit > 0) {
      const fact = 2*radiusSegments*(heightSegments+1) / faces_limit;
      if (fact > 1.0) {
         radiusSegments = Math.max(5, Math.floor(radiusSegments/Math.sqrt(fact)));
         heightSegments = Math.max(5, Math.floor(heightSegments/Math.sqrt(fact)));
      }
   }

   const rmin = shape.fRlo, rmax = shape.fRhi;
   let numfaces = (heightSegments+1) * radiusSegments*2;

   if (rmin === 0) numfaces -= radiusSegments*2; // complete layer
   if (rmax === 0) numfaces -= radiusSegments*2; // complete layer

   if (faces_limit < 0) return numfaces;

   let zmin = -shape.fDZ, zmax = shape.fDZ;

   // if no radius at -z, find intersection
   if (shape.fA >= 0)
      zmin = Math.max(zmin, shape.fB);
   else
      zmax = Math.min(shape.fB, zmax);

   const ttmin = Math.atan2(zmin, rmin),
         ttmax = Math.atan2(zmax, rmax),
         // calculate all sin/cos tables in advance
         _sin = new Float32Array(radiusSegments+1),
         _cos = new Float32Array(radiusSegments+1);
   for (let seg = 0; seg <= radiusSegments; ++seg) {
      _cos[seg] = Math.cos(seg/radiusSegments*2*Math.PI);
      _sin[seg] = Math.sin(seg/radiusSegments*2*Math.PI);
   }

   const creator = faces_limit ? new PolygonsCreator() : new GeometryCreator(numfaces);
   let lastz = zmin, lastr = 0, lastnxy = 0, lastnz = -1;

   for (let layer = 0; layer <= heightSegments + 1; ++layer) {
      if ((layer === 0) && (rmin === 0))
         continue;

      if ((layer === heightSegments + 1) && (lastr === 0))
         break;

      let layerz, radius;

      switch (layer) {
         case 0: layerz = zmin; radius = rmin; break;
         case heightSegments: layerz = zmax; radius = rmax; break;
         case heightSegments + 1: layerz = zmax; radius = 0; break;
         default: {
            const tt = Math.tan(ttmin + (ttmax-ttmin) * layer / heightSegments),
                  delta = tt**2 - 4*shape.fA*shape.fB; // should be always positive (a*b < 0)
            radius = 0.5*(tt+Math.sqrt(delta))/shape.fA;
            if (radius < 1e-6) radius = 0;
            layerz = radius*tt;
         }
      }

      const nxy = shape.fA * radius,
            nz = (shape.fA > 0) ? -1 : 1,
            skip = (lastr === 0) ? 1 : ((radius === 0) ? 2 : 0);

      for (let seg = 0; seg < radiusSegments; ++seg) {
         creator.addFace4(radius*_cos[seg], radius*_sin[seg], layerz,
                          lastr*_cos[seg], lastr*_sin[seg], lastz,
                          lastr*_cos[seg+1], lastr*_sin[seg+1], lastz,
                          radius*_cos[seg+1], radius*_sin[seg+1], layerz, skip);

         // use analytic normal values when open/closing paraboloid around 0
         // cut faces (top or bottom) set with simple normal
         if ((skip === 0) || ((layer === 1) && (rmin === 0)) || ((layer === heightSegments+1) && (rmax === 0))) {
            creator.setNormal4(nxy*_cos[seg], nxy*_sin[seg], nz,
                               lastnxy*_cos[seg], lastnxy*_sin[seg], lastnz,
                               lastnxy*_cos[seg+1], lastnxy*_sin[seg+1], lastnz,
                               nxy*_cos[seg+1], nxy*_sin[seg+1], nz, skip);
         } else
            creator.setNormal(0, 0, (layer < heightSegments) ? -1 : 1);
      }

      lastz = layerz; lastr = radius;
      lastnxy = nxy; lastnz = nz;
   }

   return creator.create();
}

/** @summary Creates hype geometry
  * @private */
function createHypeBuffer(shape, faces_limit) {
   if ((shape.fTin === 0) && (shape.fTout === 0))
      return createTubeBuffer(shape, faces_limit);

   let radiusSegments = Math.max(4, Math.round(360 / _cfg.GradPerSegm)),
       heightSegments = 30,
       numfaces = radiusSegments * (heightSegments + 1) * ((shape.fRmin > 0) ? 4 : 2);

   if (faces_limit < 0)
      return numfaces;

   if ((faces_limit > 0) && (faces_limit > numfaces)) {
      radiusSegments = Math.max(4, Math.floor(radiusSegments/Math.sqrt(numfaces/faces_limit)));
      heightSegments = Math.max(4, Math.floor(heightSegments/Math.sqrt(numfaces/faces_limit)));
      numfaces = radiusSegments * (heightSegments + 1) * ((shape.fRmin > 0) ? 4 : 2);
   }

   // calculate all sin/cos tables in advance
   const _sin = new Float32Array(radiusSegments+1),
         _cos = new Float32Array(radiusSegments+1);
   for (let seg=0; seg<=radiusSegments; ++seg) {
      _cos[seg] = Math.cos(seg/radiusSegments*2*Math.PI);
      _sin[seg] = Math.sin(seg/radiusSegments*2*Math.PI);
   }

   const creator = faces_limit ? new PolygonsCreator() : new GeometryCreator(numfaces);

   // in-out side
   for (let side = 0; side < 2; ++side) {
      if ((side > 0) && (shape.fRmin <= 0)) break;

      const r0 = (side > 0) ? shape.fRmin : shape.fRmax,
            tsq = (side > 0) ? shape.fTinsq : shape.fToutsq,
            d1 = 1- side, d2 = 1 - d1;

      // vertical layers
      for (let layer = 0; layer < heightSegments; ++layer) {
         const z1 = -shape.fDz + layer/heightSegments*2*shape.fDz,
               z2 = -shape.fDz + (layer+1)/heightSegments*2*shape.fDz,
               r1 = Math.sqrt(r0**2 + tsq*z1**2),
               r2 = Math.sqrt(r0**2 + tsq*z2**2);

         for (let seg = 0; seg < radiusSegments; ++seg) {
            creator.addFace4(r1 * _cos[seg+d1], r1 * _sin[seg+d1], z1,
                             r2 * _cos[seg+d1], r2 * _sin[seg+d1], z2,
                             r2 * _cos[seg+d2], r2 * _sin[seg+d2], z2,
                             r1 * _cos[seg+d2], r1 * _sin[seg+d2], z1);
            creator.calcNormal();
         }
      }
   }

   // add caps
   for (let layer = 0; layer < 2; ++layer) {
      const z = (layer === 0) ? shape.fDz : -shape.fDz,
            r1 = Math.sqrt(shape.fRmax**2 + shape.fToutsq*z**2),
            r2 = (shape.fRmin > 0) ? Math.sqrt(shape.fRmin**2 + shape.fTinsq*z**2) : 0,
            skip = (shape.fRmin > 0) ? 0 : 1,
            d1 = 1 - layer, d2 = 1 - d1;
      for (let seg = 0; seg < radiusSegments; ++seg) {
          creator.addFace4(r1 * _cos[seg+d1], r1 * _sin[seg+d1], z,
                           r2 * _cos[seg+d1], r2 * _sin[seg+d1], z,
                           r2 * _cos[seg+d2], r2 * _sin[seg+d2], z,
                           r1 * _cos[seg+d2], r1 * _sin[seg+d2], z, skip);
          creator.setNormal(0, 0, (layer === 0) ? 1 : -1);
       }
   }

   return creator.create();
}

/** @summary Creates tessellated geometry
  * @private */
function createTessellatedBuffer(shape, faces_limit) {
   let numfaces = 0;
   for (let i = 0; i < shape.fFacets.length; ++i)
      numfaces += (shape.fFacets[i].fNvert === 4) ? 2 : 1;
   if (faces_limit < 0) return numfaces;

   const creator = faces_limit ? new PolygonsCreator() : new GeometryCreator(numfaces);

   for (let i = 0; i < shape.fFacets.length; ++i) {
      const f = shape.fFacets[i],
            v0 = shape.fVertices[f.fIvert[0]].fVec,
            v1 = shape.fVertices[f.fIvert[1]].fVec,
            v2 = shape.fVertices[f.fIvert[2]].fVec;

      if (f.fNvert === 4) {
         const v3 = shape.fVertices[f.fIvert[3]].fVec;
         creator.addFace4(v0[0], v0[1], v0[2], v1[0], v1[1], v1[2], v2[0], v2[1], v2[2], v3[0], v3[1], v3[2]);
         creator.calcNormal();
      } else {
         creator.addFace3(v0[0], v0[1], v0[2], v1[0], v1[1], v1[2], v2[0], v2[1], v2[2]);
         creator.calcNormal();
      }
   }

   return creator.create();
}

/** @summary Creates Matrix4 from TGeoMatrix
  * @private */
function createMatrix(matrix) {
   if (!matrix) return null;

   let translation, rotation, scale;

   switch (matrix._typename) {
      case 'TGeoTranslation': translation = matrix.fTranslation; break;
      case 'TGeoRotation': rotation = matrix.fRotationMatrix; break;
      case 'TGeoScale': scale = matrix.fScale; break;
      case 'TGeoGenTrans':
         scale = matrix.fScale; // no break, translation and rotation follows
      // eslint-disable-next-line  no-fallthrough
      case 'TGeoCombiTrans':
         translation = matrix.fTranslation;
         if (matrix.fRotation) rotation = matrix.fRotation.fRotationMatrix;
         break;
      case 'TGeoHMatrix':
         translation = matrix.fTranslation;
         rotation = matrix.fRotationMatrix;
         scale = matrix.fScale;
         break;
      case 'TGeoIdentity':
         break;
      default:
         console.warn(`unsupported matrix ${matrix._typename}`);
   }

   if (!translation && !rotation && !scale) return null;

   const res = new THREE.Matrix4();

   if (rotation) {
      res.set(rotation[0], rotation[1], rotation[2], 0,
              rotation[3], rotation[4], rotation[5], 0,
              rotation[6], rotation[7], rotation[8], 0,
              0, 0, 0, 1);
   }

   if (translation)
      res.setPosition(translation[0], translation[1], translation[2]);

   if (scale)
      res.scale(new THREE.Vector3(scale[0], scale[1], scale[2]));

   return res;
}

/** @summary Creates transformation matrix for TGeoNode
  * @desc created after node visibility flag is checked and volume cut is performed
  * @private */
function getNodeMatrix(kind, node) {
   let matrix = null;

   if (kind === kindEve) {
      // special handling for EVE nodes

      matrix = new THREE.Matrix4();

      if (node.fTrans) {
         matrix.set(node.fTrans[0], node.fTrans[4], node.fTrans[8], 0,
                    node.fTrans[1], node.fTrans[5], node.fTrans[9], 0,
                    node.fTrans[2], node.fTrans[6], node.fTrans[10], 0,
                    0, 0, 0, 1);
         // second - set position with proper sign
         matrix.setPosition(node.fTrans[12], node.fTrans[13], node.fTrans[14]);
      }
   } else if (node.fMatrix)
      matrix = createMatrix(node.fMatrix);
    else if ((node._typename === 'TGeoNodeOffset') && node.fFinder) {
      const kPatternReflected = BIT(14),
            finder = node.fFinder,
            typ = finder._typename;
      if (finder.fBits & kPatternReflected)
         geoWarn(`Unsupported reflected pattern ${typ}`);
      if (typ.indexOf('TGeoPattern'))
         geoWarn(`Abnormal pattern type ${typ}`);
      const part = typ.slice(11);
      matrix = new THREE.Matrix4();
      switch (part) {
         case 'X':
         case 'Y':
         case 'Z':
         case 'ParaX':
         case 'ParaY':
         case 'ParaZ': {
            const _shift = finder.fStart + (node.fIndex + 0.5) * finder.fStep;
            switch (part.at(-1)) {
               case 'X': matrix.setPosition(_shift, 0, 0); break;
               case 'Y': matrix.setPosition(0, _shift, 0); break;
               case 'Z': matrix.setPosition(0, 0, _shift); break;
            }
            break;
         }
         case 'CylPhi': {
            const phi = (Math.PI/180)*(finder.fStart+(node.fIndex+0.5)*finder.fStep),
                 _cos = Math.cos(phi), _sin = Math.sin(phi);
            matrix.set(_cos, -_sin, 0, 0,
                       _sin, _cos, 0, 0,
                       0, 0, 1, 0,
                       0, 0, 0, 1);
            break;
         }
         case 'CylR':
            // seems to be, require no transformation
            break;
         case 'TrapZ': {
            const dz = finder.fStart + (node.fIndex+0.5)*finder.fStep;
            matrix.setPosition(finder.fTxz*dz, finder.fTyz*dz, dz);
            break;
         }
         // case 'CylR': break;
         // case 'SphR': break;
         // case 'SphTheta': break;
         // case 'SphPhi': break;
         // case 'Honeycomb': break;
         default:
            geoWarn(`Unsupported pattern type ${typ}`);
            break;
      }
   }

   return matrix;
}

/** @summary Returns number of faces for provided geometry
  * @param {Object} geom  - can be BufferGeometry, CsgGeometry or interim array of polygons
  * @private */
function numGeometryFaces(geom) {
   if (!geom) return 0;

   if (geom instanceof CsgGeometry)
      return geom.tree.numPolygons();

   // special array of polygons
   if (geom.polygons)
      return geom.polygons.length;

   const attr = geom.getAttribute('position');
   return attr?.count ? Math.round(attr.count / 3) : 0;
}

/** @summary Returns number of faces for provided geometry
  * @param {Object} geom  - can be BufferGeometry, CsgGeometry or interim array of polygons
  * @private */
function numGeometryVertices(geom) {
   if (!geom) return 0;

   if (geom instanceof CsgGeometry)
      return geom.tree.numPolygons() * 3;

   if (geom.polygons)
      return geom.polygons.length * 4;

   return geom.getAttribute('position')?.count || 0;
}

/** @summary Returns geometry bounding box
  * @private */
function geomBoundingBox(geom) {
   if (!geom) return null;

   let polygons = null;

   if (geom instanceof CsgGeometry)
      polygons = geom.tree.collectPolygons();
   else if (geom.polygons)
      polygons = geom.polygons;

   if (polygons !== null) {
      const box = new THREE.Box3();
      for (let n = 0; n < polygons.length; ++n) {
         const polygon = polygons[n], nvert = polygon.vertices.length;
         for (let k = 0; k < nvert; ++k)
            box.expandByPoint(polygon.vertices[k]);
      }
      return box;
   }

   if (!geom.boundingBox)
      geom.computeBoundingBox();

   return geom.boundingBox.clone();
}

/** @summary Creates half-space geometry for given shape
  * @desc Just big-enough triangle to make BSP calculations
  * @private */
function createHalfSpace(shape, geom) {
   if (!shape?.fN || !shape?.fP) return null;

   const vertex = new THREE.Vector3(shape.fP[0], shape.fP[1], shape.fP[2]),
         normal = new THREE.Vector3(shape.fN[0], shape.fN[1], shape.fN[2]);

   normal.normalize();

   let sz = 1e10;
   if (geom) {
      // using real size of other geometry, we probably improve precision
      const box = geomBoundingBox(geom);
      if (box) sz = box.getSize(new THREE.Vector3()).length() * 1000;
   }

   const v0 = new THREE.Vector3(-sz, -sz/2, 0),
         v1 = new THREE.Vector3(0, sz, 0),
         v2 = new THREE.Vector3(sz, -sz/2, 0),
         v3 = new THREE.Vector3(0, 0, -sz),
         geometry = new THREE.BufferGeometry(),
         positions = new Float32Array([v0.x, v0.y, v0.z, v2.x, v2.y, v2.z, v1.x, v1.y, v1.z,
                                      v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, v3.x, v3.y, v3.z,
                                      v1.x, v1.y, v1.z, v2.x, v2.y, v2.z, v3.x, v3.y, v3.z,
                                      v2.x, v2.y, v2.z, v0.x, v0.y, v0.z, v3.x, v3.y, v3.z]);
   geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
   geometry.computeVertexNormals();

   geometry.lookAt(normal);
   geometry.computeVertexNormals();

   for (let k = 0; k < positions.length; k += 3) {
      positions[k] += vertex.x;
      positions[k+1] += vertex.y;
      positions[k+2] += vertex.z;
   }

   return geometry;
}

/** @summary Returns number of faces for provided geometry
  * @param geom  - can be BufferGeometry, CsgGeometry or interim array of polygons
  * @private */
function countGeometryFaces(geom) {
   if (!geom) return 0;

   if (geom instanceof CsgGeometry)
      return geom.tree.numPolygons();

   // special array of polygons
   if (geom.polygons)
      return geom.polygons.length;

   const attr = geom.getAttribute('position');
   return attr?.count ? Math.round(attr.count / 3) : 0;
}

let createGeometry = null;

/** @summary Creates geometry for composite shape
  * @private */
function createComposite(shape, faces_limit) {
   if (faces_limit < 0) {
      return createGeometry(shape.fNode.fLeft, -1) +
             createGeometry(shape.fNode.fRight, -1);
   }

   let geom1, geom2, return_bsp = false;
   const matrix1 = createMatrix(shape.fNode.fLeftMat),
         matrix2 = createMatrix(shape.fNode.fRightMat);

   if (faces_limit === 0) faces_limit = 4000;
                     else return_bsp = true;

   if (matrix1 && (matrix1.determinant() < -0.9))
      geoWarn('Axis reflection in left composite shape - not supported');

   if (matrix2 && (matrix2.determinant() < -0.9))
      geoWarn('Axis reflections in right composite shape - not supported');

   if (shape.fNode.fLeft._typename === clTGeoHalfSpace)
      geom1 = createHalfSpace(shape.fNode.fLeft);
    else
      geom1 = createGeometry(shape.fNode.fLeft, faces_limit);

   if (!geom1) return null;

   let n1 = countGeometryFaces(geom1), n2 = 0;
   if (geom1._exceed_limit) n1 += faces_limit;

   if (n1 < faces_limit) {
      if (shape.fNode.fRight._typename === clTGeoHalfSpace)
         geom2 = createHalfSpace(shape.fNode.fRight, geom1);
       else
         geom2 = createGeometry(shape.fNode.fRight, faces_limit);


      n2 = countGeometryFaces(geom2);
   }

   if ((n1 + n2 >= faces_limit) || !geom2) {
      if (geom1.polygons)
         geom1 = createBufferGeometry(geom1.polygons);
      if (matrix1) geom1.applyMatrix4(matrix1);
      geom1._exceed_limit = true;
      return geom1;
   }

   let bsp1 = new CsgGeometry(geom1, matrix1, _cfg.CompressComp ? 0 : undefined);

   const bsp2 = new CsgGeometry(geom2, matrix2, bsp1.maxid);

   // take over maxid from both geometries
   bsp1.maxid = bsp2.maxid;

   switch (shape.fNode._typename) {
      case 'TGeoIntersection': bsp1.direct_intersect(bsp2); break; // '*'
      case 'TGeoUnion': bsp1.direct_union(bsp2); break;   // '+'
      case 'TGeoSubtraction': bsp1.direct_subtract(bsp2); break; // '/'
      default:
         geoWarn('unsupported bool operation ' + shape.fNode._typename + ', use first geom');
   }

   if (countGeometryFaces(bsp1) === 0) {
      geoWarn('Zero faces in comp shape' +
             ` left: ${shape.fNode.fLeft._typename} ${countGeometryFaces(geom1)} faces` +
             ` right: ${shape.fNode.fRight._typename} ${countGeometryFaces(geom2)} faces` +
             '  use first');
      bsp1 = new CsgGeometry(geom1, matrix1);
   }

   return return_bsp ? { polygons: bsp1.toPolygons() } : bsp1.toBufferGeometry();
}

/** @summary Try to create projected geometry
  * @private */
function projectGeometry(geom, matrix, projection, position, flippedMesh) {
   if (!geom.boundingBox) geom.computeBoundingBox();

   const box = geom.boundingBox.clone();

   box.applyMatrix4(matrix);

   if (!position) position = 0;

   if (((box.min[projection] >= position) && (box.max[projection] >= position)) ||
       ((box.min[projection] <= position) && (box.max[projection] <= position)))
      return null; // not interesting


   const bsp1 = new CsgGeometry(geom, matrix, 0, flippedMesh),
         sizex = 2*Math.max(Math.abs(box.min.x), Math.abs(box.max.x)),
         sizey = 2*Math.max(Math.abs(box.min.y), Math.abs(box.max.y)),
         sizez = 2*Math.max(Math.abs(box.min.z), Math.abs(box.max.z));
   let size = 10000;

   switch (projection) {
      case 'x': size = Math.max(sizey, sizez); break;
      case 'y': size = Math.max(sizex, sizez); break;
      case 'z': size = Math.max(sizex, sizey); break;
   }

   const bsp2 = createNormal(projection, position, size);

   bsp1.cut_from_plane(bsp2);

   return bsp2.toBufferGeometry();
}

/** @summary Creates geometry model for the provided shape
  * @param {Object} shape - instance of TGeoShape object
  * @param {Number} limit - defines return value, see details
  * @desc
  *  - if limit === 0 (or undefined) returns BufferGeometry
  *  - if limit < 0 just returns estimated number of faces
  *  - if limit > 0 return list of CsgPolygons (used only for composite shapes)
  * @private */
createGeometry = function(shape, limit) {
   if (limit === undefined) limit = 0;

   try {
      switch (shape._typename) {
         case clTGeoBBox: return createCubeBuffer(shape, limit);
         case clTGeoPara: return createParaBuffer(shape, limit);
         case clTGeoTrd1:
         case clTGeoTrd2: return createTrapezoidBuffer(shape, limit);
         case clTGeoArb8:
         case clTGeoTrap:
         case clTGeoGtra: return createArb8Buffer(shape, limit);
         case clTGeoSphere: return createSphereBuffer(shape, limit);
         case clTGeoCone:
         case clTGeoConeSeg:
         case clTGeoTube:
         case clTGeoTubeSeg:
         case clTGeoCtub: return createTubeBuffer(shape, limit);
         case clTGeoEltu: return createEltuBuffer(shape, limit);
         case clTGeoTorus: return createTorusBuffer(shape, limit);
         case clTGeoPcon:
         case clTGeoPgon: return createPolygonBuffer(shape, limit);
         case clTGeoXtru: return createXtruBuffer(shape, limit);
         case clTGeoParaboloid: return createParaboloidBuffer(shape, limit);
         case clTGeoHype: return createHypeBuffer(shape, limit);
         case 'TGeoTessellated': return createTessellatedBuffer(shape, limit);
         case clTGeoCompositeShape: return createComposite(shape, limit);
         case clTGeoShapeAssembly: break;
         case clTGeoScaledShape: {
            const res = createGeometry(shape.fShape, limit);
            if (shape.fScale && (limit >= 0) && isFunc(res?.scale))
               res.scale(shape.fScale.fScale[0], shape.fScale.fScale[1], shape.fScale.fScale[2]);
            return res;
         }
         case clTGeoHalfSpace:
            if (limit < 0)
               return 1; // half space if just plane used in composite
         // eslint-disable-next-line  no-fallthrough
         default:
            geoWarn(`unsupported shape type ${shape._typename}`);
      }
   } catch (e) {
      let place = '';
      if (e.stack !== undefined) {
         place = e.stack.split('\n')[0];
         if (place.indexOf(e.message) >= 0) place = e.stack.split('\n')[1];
                                       else place = 'at: ' + place;
      }
      geoWarn(`${shape._typename} err: ${e.message} ${place}`);
   }

   return limit < 0 ? 0 : null;
};


/** @summary Create single shape from EVE7 render date
  * @private */
function makeEveGeometry(rd) {
   let off = 0;

   if (rd.sz[0]) {
      rd.vtxBuff = new Float32Array(rd.raw.buffer, off, rd.sz[0]);
      off += rd.sz[0]*4;
   }

   if (rd.sz[1]) {
      // normals were not used
      // rd.nrmBuff = new Float32Array(rd.raw.buffer, off, rd.sz[1]);
      off += rd.sz[1]*4;
   }

   if (rd.sz[2]) {
      // these are special values in the buffer begin
      rd.prefixBuf = new Uint32Array(rd.raw.buffer, off, 2);
      off += 2*4;
      rd.idxBuff = new Uint32Array(rd.raw.buffer, off, rd.sz[2]-2);
      // off += (rd.sz[2]-2)*4;
   }

   const GL_TRIANGLES = 4; // same as in EVE7

   if (rd.prefixBuf[0] !== GL_TRIANGLES)
      throw Error('Expect triangles first.');

   const nVert = 3 * rd.prefixBuf[1]; // number of vertices to draw

   if (rd.idxBuff.length !== nVert)
      throw Error('Expect single list of triangles in index buffer.');

   const body = new THREE.BufferGeometry();
   body.setAttribute('position', new THREE.BufferAttribute(rd.vtxBuff, 3));
   body.setIndex(new THREE.BufferAttribute(rd.idxBuff, 1));
   body.computeVertexNormals();

   return body;
}

/** @summary Create single shape from geometry viewer render date
  * @private */
function makeViewerGeometry(rd) {
   const vtxBuff = new Float32Array(rd.raw.buffer, 0, rd.raw.buffer.byteLength/4),

   body = new THREE.BufferGeometry();
   body.setAttribute('position', new THREE.BufferAttribute(vtxBuff, 3));
   body.setIndex(new THREE.BufferAttribute(new Uint32Array(rd.idx), 1));
   body.computeVertexNormals();
   return body;
}

/** @summary Create single shape from provided raw data from web viewer.
  * @desc If nsegm changed, shape will be recreated
  * @private */
function createServerGeometry(rd, nsegm) {
   if (rd.server_shape && ((rd.nsegm === nsegm) || !rd.shape))
      return rd.server_shape;

   rd.nsegm = nsegm;

   let geom;

   if (rd.shape) {
      // case when TGeoShape provided as is
      geom = createGeometry(rd.shape);
   } else {
      if (!rd.raw?.buffer) {
         console.error('No raw data at all');
         return null;
      }

      geom = rd.sz ? makeEveGeometry(rd) : makeViewerGeometry(rd);
   }

   // shape handle is similar to created in TGeoPainter
   return {
      _typename: '$$Shape$$', // indicate that shape can be used as is
      ready: true,
      geom,
      nfaces: numGeometryFaces(geom)
   };
}

/** @summary Provides info about geo object, used for tooltip info
  * @param {Object} obj - any kind of TGeo-related object like shape or node or volume
  * @private */
function provideObjectInfo(obj) {
   let info = [], shape = null;

   if (obj.fVolume !== undefined)
      shape = obj.fVolume.fShape;
   else if (obj.fShape !== undefined)
      shape = obj.fShape;
   else if ((obj.fShapeBits !== undefined) && (obj.fShapeId !== undefined))
      shape = obj;

   if (!shape) {
      info.push(obj._typename);
      return info;
   }

   const sz = Math.max(shape.fDX, shape.fDY, shape.fDZ),
         useexp = (sz > 1e7) || (sz < 1e-7),
         conv = (v) => {
            if (v === undefined) return '???';
            if ((v === Math.round(v) && v < 1e7)) return Math.round(v);
            return useexp ? v.toExponential(4) : v.toPrecision(7);
         };

   info.push(shape._typename);

   info.push(`DX=${conv(shape.fDX)} DY=${conv(shape.fDY)} DZ=${conv(shape.fDZ)}`);

   switch (shape._typename) {
      case clTGeoBBox: break;
      case clTGeoPara: info.push(`Alpha=${shape.fAlpha} Phi=${shape.fPhi} Theta=${shape.fTheta}`); break;
      case clTGeoTrd2: info.push(`Dy1=${conv(shape.fDy1)} Dy2=${conv(shape.fDy1)}`); // no break
      // eslint-disable-next-line  no-fallthrough
      case clTGeoTrd1: info.push(`Dx1=${conv(shape.fDx1)} Dx2=${conv(shape.fDx1)}`); break;
      case clTGeoArb8: break;
      case clTGeoTrap: break;
      case clTGeoGtra: break;
      case clTGeoSphere:
         info.push(`Rmin=${conv(shape.fRmin)} Rmax=${conv(shape.fRmax)}`,
                   `Phi1=${shape.fPhi1} Phi2=${shape.fPhi2}`,
                   `Theta1=${shape.fTheta1} Theta2=${shape.fTheta2}`);
         break;
      case clTGeoConeSeg:
         info.push(`Phi1=${shape.fPhi1} Phi2=${shape.fPhi2}`);
      // eslint-disable-next-line  no-fallthrough
      case clTGeoCone:
         info.push(`Rmin1=${conv(shape.fRmin1)} Rmax1=${conv(shape.fRmax1)}`,
                   `Rmin2=${conv(shape.fRmin2)} Rmax2=${conv(shape.fRmax2)}`);
         break;
      case clTGeoCtub:
      case clTGeoTubeSeg:
         info.push(`Phi1=${shape.fPhi1} Phi2=${shape.fPhi2}`);
      // eslint-disable-next-line  no-fallthrough
      case clTGeoEltu:
      case clTGeoTube:
         info.push(`Rmin=${conv(shape.fRmin)} Rmax=${conv(shape.fRmax)}`);
         break;
      case clTGeoTorus:
         info.push(`Rmin=${conv(shape.fRmin)} Rmax=${conv(shape.fRmax)}`,
                   `Phi1=${shape.fPhi1} Dphi=${shape.fDphi}`);
         break;
      case clTGeoPcon:
      case clTGeoPgon: break;
      case clTGeoXtru: break;
      case clTGeoParaboloid:
         info.push(`Rlo=${conv(shape.fRlo)} Rhi=${conv(shape.fRhi)}`,
                   `A=${conv(shape.fA)} B=${conv(shape.fB)}`);
         break;
      case clTGeoHype:
         info.push(`Rmin=${conv(shape.fRmin)} Rmax=${conv(shape.fRmax)}`,
                   `StIn=${conv(shape.fStIn)} StOut=${conv(shape.fStOut)}`);
         break;
      case clTGeoCompositeShape: break;
      case clTGeoShapeAssembly: break;
      case clTGeoScaledShape:
         info = provideObjectInfo(shape.fShape);
         if (shape.fScale)
            info.unshift(`Scale X=${shape.fScale.fScale[0]} Y=${shape.fScale.fScale[1]} Z=${shape.fScale.fScale[2]}`);
         break;
   }

   return info;
}

/** @summary Creates projection matrix for the camera
  * @private */
function createProjectionMatrix(camera) {
   const cameraProjectionMatrix = new THREE.Matrix4();

   camera.updateMatrixWorld();

   camera.matrixWorldInverse.copy(camera.matrixWorld).invert();
   cameraProjectionMatrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);

   return cameraProjectionMatrix;
}

/** @summary Creates frustum
  * @private */
function createFrustum(source) {
   if (!source) return null;

   if (source instanceof THREE.PerspectiveCamera)
      source = createProjectionMatrix(source);

   const frustum = new THREE.Frustum();
   frustum.setFromProjectionMatrix(source);

   frustum.corners = new Float32Array([
       1, 1, 1,
       1, 1, -1,
       1, -1, 1,
       1, -1, -1,
      -1, 1, 1,
      -1, 1, -1,
      -1, -1, 1,
      -1, -1, -1,
       0, 0, 0 // also check center of the shape
   ]);

   frustum.test = new THREE.Vector3(0, 0, 0);

   frustum.CheckShape = function(matrix, shape) {
      const pnt = this.test, len = this.corners.length, corners = this.corners;

      for (let i = 0; i < len; i+=3) {
         pnt.x = corners[i] * shape.fDX;
         pnt.y = corners[i+1] * shape.fDY;
         pnt.z = corners[i+2] * shape.fDZ;
         if (this.containsPoint(pnt.applyMatrix4(matrix))) return true;
     }

     return false;
   };

   frustum.CheckBox = function(box) {
      const pnt = this.test;
      let cnt = 0;
      pnt.set(box.min.x, box.min.y, box.min.z);
      if (this.containsPoint(pnt)) cnt++;
      pnt.set(box.min.x, box.min.y, box.max.z);
      if (this.containsPoint(pnt)) cnt++;
      pnt.set(box.min.x, box.max.y, box.min.z);
      if (this.containsPoint(pnt)) cnt++;
      pnt.set(box.min.x, box.max.y, box.max.z);
      if (this.containsPoint(pnt)) cnt++;
      pnt.set(box.max.x, box.max.y, box.max.z);
      if (this.containsPoint(pnt)) cnt++;
      pnt.set(box.max.x, box.min.y, box.max.z);
      if (this.containsPoint(pnt)) cnt++;
      pnt.set(box.max.x, box.max.y, box.min.z);
      if (this.containsPoint(pnt)) cnt++;
      pnt.set(box.max.x, box.max.y, box.max.z);
      if (this.containsPoint(pnt)) cnt++;
      return cnt > 5; // only if 6 edges and more are seen, we think that box is fully visible
   };

   return frustum;
}

/** @summary Create node material
  * @private */
function createMaterial(cfg, args0) {
   if (!cfg) cfg = { material_kind: 'lambert' };

   const args = Object.assign({}, args0);

   if (args.opacity === undefined)
      args.opacity = 1;

   if (cfg.transparency)
      args.opacity = Math.min(1 - cfg.transparency, args.opacity);

   args.wireframe = cfg.wireframe ?? false;
   if (!args.color) args.color = 'red';
   args.side = THREE.FrontSide;
   args.transparent = args.opacity < 1;
   args.depthWrite = args.opactity === 1;

   let material;

   if (cfg.material_kind === 'basic')
      material = new THREE.MeshBasicMaterial(args);
    else if (cfg.material_kind === 'depth') {
      delete args.color;
      material = new THREE.MeshDepthMaterial(args);
   } else if (cfg.material_kind === 'toon')
      material = new THREE.MeshToonMaterial(args);
    else if (cfg.material_kind === 'matcap') {
      delete args.wireframe;
      material = new THREE.MeshMatcapMaterial(args);
   } else if (cfg.material_kind === 'standard') {
      args.metalness = cfg.metalness ?? 0.5;
      args.roughness = cfg.roughness ?? 0.1;
      material = new THREE.MeshStandardMaterial(args);
   } else if (cfg.material_kind === 'normal') {
      delete args.color;
      material = new THREE.MeshNormalMaterial(args);
   } else if (cfg.material_kind === 'physical') {
      args.metalness = cfg.metalness ?? 0.5;
      args.roughness = cfg.roughness ?? 0.1;
      args.reflectivity = cfg.reflectivity ?? 0.5;
      args.emissive = args.color;
      material = new THREE.MeshPhysicalMaterial(args);
   } else if (cfg.material_kind === 'phong') {
      args.shininess = cfg.shininess ?? 0.9;
      material = new THREE.MeshPhongMaterial(args);
   } else {
      args.vertexColors = false;
      material = new THREE.MeshLambertMaterial(args);
   }

   if ((material.flatShading !== undefined) && (cfg.flatShading !== undefined))
      material.flatShading = cfg.flatShading;
   material.inherentOpacity = args0.opacity ?? 1;
   material.inherentArgs = args0;

   return material;
}

/** @summary Compares two stacks.
  * @return {Number} 0 if same, -1 when stack1 < stack2, +1 when stack1 > stack2
  * @private */
function compare_stacks(stack1, stack2) {
   if (stack1 === stack2)
      return 0;
   const len1 = stack1?.length ?? 0,
         len2 = stack2?.length ?? 0,
         len = (len1 < len2) ? len1 : len2;
   let indx = 0;
   while (indx < len) {
      if (stack1[indx] < stack2[indx])
         return -1;
      if (stack1[indx] > stack2[indx])
         return 1;
      ++indx;
   }

   return (len1 < len2) ? -1 : ((len1 > len2) ? 1 : 0);
}

/** @summary Checks if two stack arrays are identical
  * @private */
function isSameStack(stack1, stack2) {
   if (!stack1 || !stack2) return false;
   if (stack1 === stack2) return true;
   if (stack1.length !== stack2.length) return false;
   for (let k = 0; k < stack1.length; ++k)
      if (stack1[k] !== stack2[k]) return false;
   return true;
}


function createFlippedGeom(geom) {
   let pos = geom.getAttribute('position').array,
       norm = geom.getAttribute('normal').array;
   const index = geom.getIndex();

   if (index) {
      // we need to unfold all points to
      const arr = index.array,
            i0 = geom.drawRange.start;
      let ilen = geom.drawRange.count;
      if (i0 + ilen > arr.length) ilen = arr.length - i0;

      const dpos = new Float32Array(ilen*3), dnorm = new Float32Array(ilen*3);
      for (let ii = 0; ii < ilen; ++ii) {
         const k = arr[i0 + ii];
         if ((k < 0) || (k*3 >= pos.length))
            console.log(`strange index ${k*3} totallen = ${pos.length}`);
         dpos[ii*3] = pos[k*3];
         dpos[ii*3+1] = pos[k*3+1];
         dpos[ii*3+2] = pos[k*3+2];
         dnorm[ii*3] = norm[k*3];
         dnorm[ii*3+1] = norm[k*3+1];
         dnorm[ii*3+2] = norm[k*3+2];
      }

      pos = dpos; norm = dnorm;
   }

   const len = pos.length,
         newpos = new Float32Array(len),
         newnorm = new Float32Array(len);

   // we should swap second and third point in each face
   for (let n = 0, shift = 0; n < len; n += 3) {
      newpos[n] = pos[n+shift];
      newpos[n+1] = pos[n+1+shift];
      newpos[n+2] = -pos[n+2+shift];

      newnorm[n] = norm[n+shift];
      newnorm[n+1] = norm[n+1+shift];
      newnorm[n+2] = -norm[n+2+shift];

      shift+=3; if (shift===6) shift=-3; // values 0,3,-3
   }

   const geomZ = new THREE.BufferGeometry();
   geomZ.setAttribute('position', new THREE.BufferAttribute(newpos, 3));
   geomZ.setAttribute('normal', new THREE.BufferAttribute(newnorm, 3));

   return geomZ;
}


/** @summary Create flipped mesh for the shape
  * @desc When transformation matrix includes one or several inversion of axis,
  * one should inverse geometry object, otherwise three.js cannot correctly draw it
  * @param {Object} shape - TGeoShape object
  * @param {Object} material - material
  * @private */
function createFlippedMesh(shape, material) {
   if (shape.geomZ === undefined)
      shape.geomZ = createFlippedGeom(shape.geom);

   const mesh = new THREE.Mesh(shape.geomZ, material);
   mesh.scale.copy(new THREE.Vector3(1, 1, -1));
   mesh.updateMatrix();

   mesh._flippedMesh = true;

   return mesh;
}


/**
  * @summary class for working with cloned nodes
  *
  * @private
  */

class ClonedNodes {

   /** @summary Constructor */
   constructor(obj, clones) {
      this.toplevel = true; // indicate if object creates top-level structure with Nodes and Volumes folder
      this.name_prefix = ''; // name prefix used for nodes names
      this.maxdepth = 1;  // maximal hierarchy depth, required for transparency
      this.vislevel = 4;  // maximal depth of nodes visibility aka gGeoManager->SetVisLevel, same default
      this.maxnodes = 10000; // maximal number of visible nodes aka gGeoManager->fMaxVisNodes

      if (obj) {
         if (obj.$geoh) this.toplevel = false;
         this.createClones(obj);
      } else if (clones)
         this.nodes = clones;
   }

   /** @summary Set maximal depth for nodes visibility */
   setVisLevel(lvl) {
      this.vislevel = lvl && Number.isInteger(lvl) ? lvl : 4;
   }

   /** @summary Returns maximal depth for nodes visibility */
   getVisLevel() {
      return this.vislevel;
   }

   /** @summary Set maximal number of visible nodes
    * @desc By default 10000 nodes will be visualized */
   setMaxVisNodes(v, more) {
      this.maxnodes = (v === Infinity) ? 1e9 : (Number.isFinite(v) ? v : 10000);
      if (more && Number.isFinite(more))
         this.maxnodes *= more;
   }

   /** @summary Returns configured maximal number of visible nodes */
   getMaxVisNodes() {
      return this.maxnodes;
   }

   /** @summary Set geo painter configuration - used for material creation */
   setConfig(cfg) {
      this._cfg = cfg;
   }

   /** @summary Insert node into existing array */
   updateNode(node) {
      if (node && Number.isInteger(node.id) && (node.id < this.nodes.length))
         this.nodes[node.id] = node;
   }

   /** @summary Returns TGeoShape for element with given indx */
   getNodeShape(indx) {
      if (!this.origin || !this.nodes) return null;
      const obj = this.origin[indx], clone = this.nodes[indx];
      if (!obj || !clone) return null;
      if (clone.kind === kindGeo) {
         if (obj.fVolume) return obj.fVolume.fShape;
      } else
         return obj.fShape;

      return null;
   }

   /** @summary function to cleanup as much as possible structures
     * @desc Provided parameters drawnodes and drawshapes are arrays created during building of geometry */
   cleanup(drawnodes, drawshapes) {
      if (drawnodes) {
         for (let n = 0; n < drawnodes.length; ++n) {
            delete drawnodes[n].stack;
            drawnodes[n] = undefined;
         }
      }

      if (drawshapes) {
         for (let n = 0; n < drawshapes.length; ++n) {
            delete drawshapes[n].geom;
            drawshapes[n] = undefined;
         }
      }

      if (this.nodes) {
         for (let n = 0; n < this.nodes.length; ++n) {
            if (this.nodes[n])
               delete this.nodes[n].chlds;
         }
      }

      delete this.nodes;
      delete this.origin;

      delete this.sortmap;
   }

   /** @summary Create complete description for provided Geo object */
   createClones(obj, sublevel, kind) {
      if (!sublevel) {
         if (obj?._typename === '$$Shape$$')
            return this.createClonesForShape(obj);

         this.origin = [];
         sublevel = 1;
         kind = getNodeKind(obj);
      }

      if ((kind < 0) || !obj || ('_refid' in obj)) return;

      obj._refid = this.origin.length;
      this.origin.push(obj);
      if (sublevel > this.maxdepth) this.maxdepth = sublevel;

      let chlds;
      if (kind === kindGeo)
         chlds = obj.fVolume?.fNodes?.arr || null;
      else
         chlds = obj.fElements?.arr || null;

      if (chlds !== null) {
         checkDuplicates(obj, chlds);
         for (let i = 0; i < chlds.length; ++i)
            this.createClones(chlds[i], sublevel + 1, kind);
      }

      if (sublevel > 1)
         return;

      this.nodes = [];

      const sortarr = [];

      // first create nodes objects
      for (let id = 0; id < this.origin.length; ++id) {
         // let obj = this.origin[id];
         const node = { id, kind, vol: 0, nfaces: 0 };
         this.nodes.push(node);
         sortarr.push(node); // array use to produce sortmap
      }

      // than fill children lists
      for (let n = 0; n < this.origin.length; ++n) {
         const obj2 = this.origin[n],
               clone = this.nodes[n],
               shape = kind === kindEve ? obj2.fShape : obj2.fVolume.fShape,
               chlds2 = kind === kindEve ? obj2.fElements?.arr : obj2.fVolume.fNodes?.arr,
               matrix = getNodeMatrix(kind, obj2);

         if (matrix) {
            clone.matrix = matrix.elements; // take only matrix elements, matrix will be constructed in worker
            if (clone.matrix && (clone.matrix[0] === 1)) {
               let issimple = true;
               for (let k = 1; (k < clone.matrix.length) && issimple; ++k)
                  issimple = (clone.matrix[k] === ((k === 5) || (k === 10) || (k === 15) ? 1 : 0));
               if (issimple) delete clone.matrix;
            }
            if (clone.matrix && (kind === kindEve))
               clone.abs_matrix = true;
         }
         if (shape) {
            clone.fDX = shape.fDX;
            clone.fDY = shape.fDY;
            clone.fDZ = shape.fDZ;
            clone.vol = Math.sqrt(shape.fDX**2 + shape.fDY**2 + shape.fDZ**2);
            if (shape.$nfaces === undefined)
               shape.$nfaces = createGeometry(shape, -1);
            clone.nfaces = shape.$nfaces;
            if (clone.nfaces <= 0)
               clone.vol = 0;
         }

         if (chlds2) {
            // in cloned object children is only list of ids
            clone.chlds = new Array(chlds2.length);
            for (let k = 0; k < chlds2.length; ++k)
               clone.chlds[k] = chlds2[k]._refid;
         }
      }

      // remove _refid identifiers from original objects
      for (let n = 0; n < this.origin.length; ++n)
         delete this.origin[n]._refid;

      // do sorting once
      sortarr.sort((a, b) => b.vol - a.vol);

      // remember sort map and also sortid
      this.sortmap = new Array(this.nodes.length);
      for (let n = 0; n < this.nodes.length; ++n) {
         this.sortmap[n] = sortarr[n].id;
         sortarr[n].sortid = n;
      }
   }

   /** @summary Create elementary item with single already existing shape
     * @desc used by details view of geometry shape */
   createClonesForShape(obj) {
      this.origin = [];

      // indicate that just plain shape is used
      this.plain_shape = obj;

      this.nodes = [{
         id: 0, sortid: 0, kind: kindShape,
         name: 'Shape',
         nfaces: obj.nfaces,
         fDX: 1, fDY: 1, fDZ: 1, vol: 1,
         vis: true
      }];
   }

   /** @summary Count all visible nodes */
   countVisibles() {
      const len = this.nodes?.length || 0;
      let cnt = 0;
      for (let k = 0; k < len; ++k)
         if (this.nodes[k].vis) cnt++;
      return cnt;
   }

   /** @summary Mark visible nodes.
     * @desc Set only basic flags, actual visibility depends from hierarchy */
   markVisibles(on_screen, copy_bits, hide_top_volume) {
      if (this.plain_shape)
         return 1;
      if (!this.origin || !this.nodes)
         return 0;

      let res = 0;

      for (let n = 0; n < this.nodes.length; ++n) {
         const clone = this.nodes[n], obj = this.origin[n];

         clone.vis = 0; // 1 - only with last level
         delete clone.nochlds;

         if (clone.kind === kindGeo) {
            if (obj.fVolume) {
               if (on_screen) {
                  // on screen bits used always, childs always checked
                  clone.vis = testGeoBit(obj.fVolume, geoBITS.kVisOnScreen) ? 99 : 0;

                  if ((n === 0) && clone.vis && hide_top_volume) clone.vis = 0;

                  if (copy_bits) {
                     setGeoBit(obj.fVolume, geoBITS.kVisNone, false);
                     setGeoBit(obj.fVolume, geoBITS.kVisThis, (clone.vis > 0));
                     setGeoBit(obj.fVolume, geoBITS.kVisDaughters, true);
                     setGeoBit(obj, geoBITS.kVisDaughters, true);
                  }
               } else {
                  clone.vis = !testGeoBit(obj.fVolume, geoBITS.kVisNone) && testGeoBit(obj.fVolume, geoBITS.kVisThis) ? 99 : 0;

                  if (!testGeoBit(obj, geoBITS.kVisDaughters) || !testGeoBit(obj.fVolume, geoBITS.kVisDaughters))
                     clone.nochlds = true;

                  // node with childs only shown in case if it is last level in hierarchy
                  if ((clone.vis > 0) && clone.chlds && !clone.nochlds)
                     clone.vis = 1;

                  // special handling for top node
                  if (n === 0) {
                     if (hide_top_volume) clone.vis = 0;
                     delete clone.nochlds;
                  }
               }
            }
         } else {
            clone.vis = obj.fRnrSelf ? 99 : 0;

            // when the only node is selected, draw it
            if ((n === 0) && (this.nodes.length === 1)) clone.vis = 99;

            this.vislevel = 9999; // automatically take all volumes
         }

         // shape with zero volume or without faces will not be observed
         if ((clone.vol <= 0) || (clone.nfaces <= 0)) clone.vis = 0;

         if (clone.vis) res++;
      }

      return res;
   }

   /** @summary After visibility flags is set, produce id shifts for all nodes as it would be maximum level */
   produceIdShifts() {
      for (let k = 0; k < this.nodes.length; ++k)
         this.nodes[k].idshift = -1;

      function scan_func(nodes, node) {
         if (node.idshift < 0) {
            node.idshift = 0;
            if (node.chlds) {
               for (let k = 0; k<node.chlds.length; ++k)
                  node.idshift += scan_func(nodes, nodes[node.chlds[k]]);
            }
         }

         return node.idshift + 1;
      }

      scan_func(this.nodes, this.nodes[0]);
   }

   /** @summary Extract only visibility flags
     * @desc Used to transfer them to the worker */
   getVisibleFlags() {
      const res = new Array(this.nodes.length);
      for (let n=0; n<this.nodes.length; ++n)
         res[n] = { vis: this.nodes[n].vis, nochlds: this.nodes[n].nochlds };
      return res;
   }

   /** @summary Assign only visibility flags, extracted with getVisibleFlags */
   setVisibleFlags(flags) {
      if (!this.nodes || !flags || !flags.length !== this.nodes.length)
         return 0;

      let res = 0;
      for (let n = 0; n < this.nodes.length; ++n) {
         const clone = this.nodes[n];
         clone.vis = flags[n].vis;
         clone.nochlds = flags[n].nochlds;
         if (clone.vis) res++;
      }

      return res;
   }

   /** @summary Set visibility flag for physical node
     * @desc Trying to reimplement functionality in the RGeomViewer */
   setPhysNodeVisibility(stack, on) {
      let do_clear = false;
      if (on === 'clearall') {
         delete this.fVisibility;
         return;
      } else if (on === 'clear') {
         do_clear = true;
         if (!this.fVisibility) return;
      } else
         on = Boolean(on);
      if (!stack)
         return;

      if (!this.fVisibility)
         this.fVisibility = [];

      for (let indx = 0; indx < this.fVisibility.length; ++indx) {
         const item = this.fVisibility[indx],
             res = compare_stacks(item.stack, stack);

         if (res === 0) {
            if (do_clear) {
               this.fVisibility.splice(indx, 1);
               if (!this.fVisibility.length)
                  delete this.fVisibility;
            } else
               item.visible = on;

            return;
         }

         if (res > 0) {
            if (!do_clear)
               this.fVisibility.splice(indx, 0, { visible: on, stack });
            return;
         }
      }

      if (!do_clear)
         this.fVisibility.push({ visible: on, stack });
   }

   /** @summary Get visibility item for physical node */
   getPhysNodeVisibility(stack) {
      if (!stack || !this.fVisibility)
         return null;
      for (let indx = 0; indx < this.fVisibility.length; ++indx) {
         const item = this.fVisibility[indx],
               res = compare_stacks(item.stack, stack);
         if (res === 0)
            return item;
         if (res > 0)
            return null;
      }

      return null;
   }

   /** @summary Scan visible nodes in hierarchy, starting from nodeid
     * @desc Each entry in hierarchy get its unique id, which is not changed with visibility flags */
   scanVisible(arg, vislvl) {
      if (!this.nodes) return 0;

      if (vislvl === undefined) {
         if (!arg) arg = {};

         vislvl = arg.vislvl || this.vislevel || 4; // default 3 in ROOT
         if (vislvl > 88) vislvl = 88;

         arg.stack = new Array(100); // current stack
         arg.nodeid = 0;
         arg.counter = 0; // sequence ID of the node, used to identify it later
         arg.last = 0;
         arg.copyStack = function(factor) {
            const entry = { nodeid: this.nodeid, seqid: this.counter, stack: new Array(this.last) };
            if (factor) entry.factor = factor; // factor used to indicate importance of entry, will be built as first
            for (let n = 0; n < this.last; ++n)
               entry.stack[n] = this.stack[n+1]; // copy stack
            return entry;
         };

         if (arg.domatrix) {
            arg.matrices = [];
            arg.mpool = [new THREE.Matrix4()]; // pool of Matrix objects to avoid permanent creation
            arg.getmatrix = function() { return this.matrices[this.last]; };
         }

         if (this.fVisibility?.length) {
            arg.vindx = 0;
            arg.varray = this.fVisibility;
            arg.vstack = arg.varray[arg.vindx].stack;
            arg.testPhysVis = function() {
               if (!this.vstack || (this.vstack?.length !== this.last))
                  return undefined;
               for (let n = 0; n < this.last; ++n) {
                  if (this.vstack[n] !== this.stack[n+1])
                     return undefined;
               }
               const res = this.varray[this.vindx++].visible;
               this.vstack = this.vindx < this.varray.length ? this.varray[this.vindx].stack : null;
               return res;
            };
         }
      }

      const node = this.nodes[arg.nodeid];
      let res = 0;

      if (arg.domatrix) {
         if (!arg.mpool[arg.last+1])
            arg.mpool[arg.last+1] = new THREE.Matrix4();

         const prnt = (arg.last > 0) ? arg.matrices[arg.last-1] : new THREE.Matrix4();
         if (node.matrix) {
            arg.matrices[arg.last] = arg.mpool[arg.last].fromArray(prnt.elements);
            arg.matrices[arg.last].multiply(arg.mpool[arg.last+1].fromArray(node.matrix));
         } else
            arg.matrices[arg.last] = prnt;
      }

      let node_vis = node.vis, node_nochlds = node.nochlds;

      if ((arg.nodeid === 0) && arg.main_visible)
         node_vis = vislvl + 1;
      else if (arg.testPhysVis) {
         const res2 = arg.testPhysVis();
         if (res2 !== undefined) {
            node_vis = res2 && !node.chlds ? vislvl + 1 : 0;
            node_nochlds = !res2;
         }
      }

      if (node_nochlds)
         vislvl = 0;

      if (node_vis > vislvl) {
         if (!arg.func || arg.func(node))
            res++;
      }

      arg.counter++;

      if ((vislvl > 0) && node.chlds) {
         arg.last++;
         for (let i = 0; i < node.chlds.length; ++i) {
            arg.nodeid = node.chlds[i];
            arg.stack[arg.last] = i; // in the stack one store index of child, it is path in the hierarchy
            res += this.scanVisible(arg, vislvl-1);
         }
         arg.last--;
      } else
         arg.counter += (node.idshift || 0);


      if (arg.last === 0) {
         delete arg.last;
         delete arg.stack;
         delete arg.copyStack;
         delete arg.counter;
         delete arg.matrices;
         delete arg.mpool;
         delete arg.getmatrix;
         delete arg.vindx;
         delete arg.varray;
         delete arg.vstack;
         delete arg.testPhysVis;
      }

      return res;
   }

   /** @summary Return node name with given id.
    * @desc Either original object or description is used */
   getNodeName(nodeid) {
      if (this.origin) {
         const obj = this.origin[nodeid];
         return obj ? getObjectName(obj) : '';
      }
      const node = this.nodes[nodeid];
      return node ? node.name : '';
   }

   /** @summary Returns description for provided stack
     * @desc If specified, absolute matrix is also calculated */
   resolveStack(stack, withmatrix) {
      const res = { id: 0, obj: null, node: this.nodes[0], name: this.name_prefix || '' };

      if (withmatrix) {
         res.matrix = new THREE.Matrix4();
         if (res.node.matrix) res.matrix.fromArray(res.node.matrix);
      }

      if (this.origin)
         res.obj = this.origin[0];

      // if (!res.name)
      //   res.name = this.getNodeName(0);

      if (stack) {
         for (let lvl = 0; lvl < stack.length; ++lvl) {
            res.id = res.node.chlds[stack[lvl]];
            res.node = this.nodes[res.id];

            if (this.origin)
               res.obj = this.origin[res.id];

            const subname = this.getNodeName(res.id);
            if (subname) {
               if (res.name) res.name += '/';
               res.name += subname;
            }

            if (withmatrix && res.node.matrix)
               res.matrix.multiply(new THREE.Matrix4().fromArray(res.node.matrix));
         }
      }

      return res;
   }

   /** @summary Provide stack name
     * @desc Stack name includes full path to the physical node which is identified by stack  */
   getStackName(stack) {
      return this.resolveStack(stack).name;
   }

   /** @summary Create stack array based on nodes ids array.
    * @desc Ids list should correspond to existing nodes hierarchy */
   buildStackByIds(ids) {
      if (!ids)
         return null;

      if (ids[0]) {
         console.error('wrong ids - first should be 0');
         return null;
      }

      let node = this.nodes[0];
      const stack = [];

      for (let k = 1; k < ids.length; ++k) {
         const nodeid = ids[k];
         if (!node) return null;
         const chindx = node.chlds.indexOf(nodeid);
         if (chindx < 0) {
            console.error(`wrong nodes ids ${ids[k]} is not child of ${ids[k-1]}`);
            return null;
         }

         stack.push(chindx);
         node = this.nodes[nodeid];
      }

      return stack;
   }

   /** @summary Returns ids array which correspond to the stack */
   buildIdsByStack(stack) {
      if (!stack)
         return null;
      let node = this.nodes[0];
      const ids = [0];
      for (let k = 0; k < stack.length; ++k) {
         const id = node.chlds[stack[k]];
         ids.push(id);
         node = this.nodes[id];
      }
      return ids;
   }

   /** @summary Returns node id by stack */
   getNodeIdByStack(stack) {
      if (!stack || !this.nodes)
         return -1;
      let node = this.nodes[0], id = 0;
      for (let k = 0; k < stack.length; ++k) {
         id = node.chlds[stack[k]];
         node = this.nodes[id];
      }
      return id;
   }

   /** @summary Returns true if stack includes at any place provided nodeid */
   isIdInStack(nodeid, stack) {
      if (!nodeid)
         return true;

      let node = this.nodes[0];

      for (let lvl = 0; lvl < stack.length; ++lvl) {
         const id = node.chlds[stack[lvl]];
         if (id === nodeid)
            return true;
         node = this.nodes[id];
      }

      return false;
   }

   /** @summary Find stack by name which include names of all parents */
   findStackByName(fullname) {
      const names = fullname.split('/'), stack = [];
      let currid = 0;

      if (this.getNodeName(currid) !== names[0])
         return null;

      for (let n = 1; n < names.length; ++n) {
         const node = this.nodes[currid];
         if (!node.chlds) return null;

         for (let k = 0; k < node.chlds.length; ++k) {
            const chldid = node.chlds[k];
            if (this.getNodeName(chldid) === names[n]) {
               stack.push(k);
               currid = chldid;
               break;
            }
         }

         // no new entry - not found stack
         if (stack.length === n - 1)
            return null;
      }

      return stack;
   }

   /** @summary Set usage of default ROOT colors */
   setDefaultColors(on) {
      this.use_dflt_colors = on;
      if (this.use_dflt_colors && !this.dflt_table) {
         const nmax = 110, col = [], dflt = { kWhite: 0, kBlack: 1, kGray: 920,
               kRed: 632, kGreen: 416, kBlue: 600, kYellow: 400, kMagenta: 616, kCyan: 432,
               kOrange: 800, kSpring: 820, kTeal: 840, kAzure: 860, kViolet: 880, kPink: 900 };
         for (let i = 0; i < nmax; i++)
            col.push(dflt.kGray);

         //  here we should create a new TColor with the same rgb as in the default
         //  ROOT colors used below
         col[3] = dflt.kYellow-10;
         col[4] = col[5] = dflt.kGreen-10;
         col[6] = col[7] = dflt.kBlue-7;
         col[8] = col[9] = dflt.kMagenta-3;
         col[10] = col[11] = dflt.kRed-10;
         col[12] = dflt.kGray+1;
         col[13] = dflt.kBlue-10;
         col[14] = dflt.kOrange+7;
         col[16] = dflt.kYellow+1;
         col[20] = dflt.kYellow-10;
         col[24] = col[25] = col[26] = dflt.kBlue-8;
         col[29] = dflt.kOrange+9;
         col[79] = dflt.kOrange-2;

         this.dflt_table = col;
      }
   }

   /** @summary Provide different properties of draw entry nodeid
     * @desc Only if node visible, material will be created */
   getDrawEntryProperties(entry, root_colors) {
      const clone = this.nodes[entry.nodeid], visible = true;

      if (clone.kind === kindShape) {
         const prop = { name: clone.name, nname: clone.name, shape: null, material: null, chlds: null },
             opacity = entry.opacity || 1, col = entry.color || '#0000FF';
         prop.fillcolor = new THREE.Color(col[0] === '#' ? col : `rgb(${col})`);
         prop.material = createMaterial(this._cfg, { opacity, color: prop.fillcolor });
         return prop;
      }

      if (!this.origin) {
         console.error(`origin not there - kind ${clone.kind} id ${entry.nodeid}`);
         return null;
      }

      const node = this.origin[entry.nodeid];

      if (clone.kind === kindEve) {
         // special handling for EVE nodes

         const prop = { name: getObjectName(node), nname: getObjectName(node), shape: node.fShape, material: null, chlds: null };

         if (node.fElements !== null) prop.chlds = node.fElements.arr;

         if (visible) {
            const opacity = Math.min(1, node.fRGBA[3]);
            prop.fillcolor = new THREE.Color(node.fRGBA[0], node.fRGBA[1], node.fRGBA[2]);
            prop.material = createMaterial(this._cfg, { opacity, color: prop.fillcolor });
         }

         return prop;
      }

      const volume = node.fVolume,
            prop = { name: getObjectName(volume), nname: getObjectName(node), volume, shape: volume.fShape, material: null,
                     chlds: volume.fNodes?.arr, linewidth: volume.fLineWidth };

      if (visible) {
         // TODO: maybe correctly extract ROOT colors here?
         let opacity = 1.0;
         if (!root_colors) root_colors = ['white', 'black', 'red', 'green', 'blue', 'yellow', 'magenta', 'cyan'];

         if (entry.custom_color)
            prop.fillcolor = entry.custom_color;
         else if ((volume.fFillColor > 1) && (volume.fLineColor === 1))
            prop.fillcolor = root_colors[volume.fFillColor];
         else if (volume.fLineColor >= 0)
            prop.fillcolor = root_colors[volume.fLineColor];

         const mat = volume.fMedium?.fMaterial;

         if (mat) {
            const fillstyle = mat.fFillStyle;
            let transparency = (fillstyle >= 3000 && fillstyle <= 3100) ? fillstyle - 3000 : 0;

            if (this.use_dflt_colors) {
               const matZ = Math.round(mat.fZ), icol = this.dflt_table[matZ];
               prop.fillcolor = root_colors[icol];
               if (mat.fDensity < 0.1) transparency = 60;
            }

            if (transparency > 0)
               opacity = (100 - transparency) / 100;
            if (prop.fillcolor === undefined)
               prop.fillcolor = root_colors[mat.fFillColor];
         }
         if (prop.fillcolor === undefined)
            prop.fillcolor = 'lightgrey';

         prop.material = createMaterial(this._cfg, { opacity, color: prop.fillcolor });
      }

      return prop;
   }

   /** @summary Creates hierarchy of Object3D for given stack entry
     * @desc Such hierarchy repeats hierarchy of TGeoNodes and set matrix for the objects drawing
     * also set renderOrder, required to handle transparency */
   createObject3D(stack, toplevel, options) {
      let node = this.nodes[0], three_prnt = toplevel, draw_depth = 0;
      const force = isObject(options) || (options === 'force');

      for (let lvl = 0; lvl <= stack.length; ++lvl) {
         const nchld = (lvl > 0) ? stack[lvl-1] : 0,
               // extract current node
               child = (lvl > 0) ? this.nodes[node.chlds[nchld]] : node;
         if (!child) {
            console.error(`Wrong stack ${JSON.stringify(stack)} for nodes at level ${lvl}, node.id ${node.id}, numnodes ${this.nodes.length}, nchld ${nchld}, numchilds ${node.chlds.length}, chldid ${node.chlds[nchld]}`);
            return null;
         }

         node = child;

         let obj3d;

         if (three_prnt.children) {
            for (let i = 0; i < three_prnt.children.length; ++i) {
               if (three_prnt.children[i].nchld === nchld) {
                  obj3d = three_prnt.children[i];
                  break;
               }
            }
         }

         if (obj3d) {
            three_prnt = obj3d;
            if (obj3d.$jsroot_drawable) draw_depth++;
            continue;
         }

         if (!force) return null;

         obj3d = new THREE.Object3D();

         if (this._cfg?.set_names)
            obj3d.name = this.getNodeName(node.id);

         if (this._cfg?.set_origin && this.origin)
            obj3d.userData = this.origin[node.id];

         if (node.abs_matrix) {
            obj3d.absMatrix = new THREE.Matrix4();
            obj3d.absMatrix.fromArray(node.matrix);
         } else if (node.matrix) {
            obj3d.matrix.fromArray(node.matrix);
            obj3d.matrix.decompose(obj3d.position, obj3d.quaternion, obj3d.scale);
         }

         // this.accountNodes(obj3d);
         obj3d.nchld = nchld; // mark index to find it again later

         // add the mesh to the scene
         three_prnt.add(obj3d);

         // this is only for debugging - test inversion of whole geometry
         if ((lvl === 0) && isObject(options) && options.scale) {
            if ((options.scale.x < 0) || (options.scale.y < 0) || (options.scale.z < 0)) {
               obj3d.scale.copy(options.scale);
               obj3d.updateMatrix();
            }
         }

         obj3d.updateMatrixWorld();

         three_prnt = obj3d;
      }

      if ((options === 'mesh') || (options === 'delete_mesh')) {
         let mesh = null;
         if (three_prnt) {
            for (let n = 0; (n < three_prnt.children.length) && !mesh; ++n) {
               const chld = three_prnt.children[n];
               if ((chld.type === 'Mesh') && (chld.nchld === undefined)) mesh = chld;
            }
         }

         if ((options === 'mesh') || !mesh)
            return mesh;

         const res = three_prnt;
         while (mesh && (mesh !== toplevel)) {
            three_prnt = mesh.parent;
            three_prnt.remove(mesh);
            mesh = !three_prnt.children.length ? three_prnt : null;
         }

         return res;
      }

      if (three_prnt) {
         three_prnt.$jsroot_drawable = true;
         three_prnt.$jsroot_depth = draw_depth;
      }

      return three_prnt;
   }

   /** @summary Create mesh for single physical node */
   createEntryMesh(ctrl, toplevel, entry, shape, colors) {
      if (!shape || !shape.ready)
         return null;

      entry.done = true; // mark entry is created
      shape.used = true; // indicate that shape was used in building

      if (!shape.geom || !shape.nfaces) {
         // node is visible, but shape does not created
         this.createObject3D(entry.stack, toplevel, 'delete_mesh');
         return null;
      }

      const prop = this.getDrawEntryProperties(entry, colors),
            obj3d = this.createObject3D(entry.stack, toplevel, ctrl),
            matrix = obj3d.absMatrix || obj3d.matrixWorld;

      prop.material.wireframe = ctrl.wireframe;

      prop.material.side = ctrl.doubleside ? THREE.DoubleSide : THREE.FrontSide;

      let mesh;
      if (matrix.determinant() > -0.9)
         mesh = new THREE.Mesh(shape.geom, prop.material);
       else
         mesh = createFlippedMesh(shape, prop.material);

      obj3d.add(mesh);

      if (obj3d.absMatrix) {
         mesh.matrix.copy(obj3d.absMatrix);
         mesh.matrix.decompose(mesh.position, mesh.quaternion, mesh.scale);
         mesh.updateMatrixWorld();
      }

      // keep full stack of nodes
      mesh.stack = entry.stack;
      mesh.renderOrder = this.maxdepth - entry.stack.length; // order of transparency handling

      if (ctrl.set_names)
         mesh.name = this.getNodeName(entry.nodeid);

      if (ctrl.set_origin)
         mesh.userData = prop.volume;

      // keep hierarchy level
      mesh.$jsroot_order = obj3d.$jsroot_depth;

      if (ctrl.info?.num_meshes !== undefined) {
         ctrl.info.num_meshes++;
         ctrl.info.num_faces += shape.nfaces;
      }

      // set initial render order, when camera moves, one must refine it
      // mesh.$jsroot_order = mesh.renderOrder =
      //   this._clones.maxdepth - ((obj3d.$jsroot_depth !== undefined) ? obj3d.$jsroot_depth : entry.stack.length);

      return mesh;
   }

   /** @summary Check if instancing can be used for the nodes */
   createInstancedMeshes(ctrl, toplevel, draw_nodes, build_shapes, colors) {
      if (ctrl.instancing < 0)
         return false;

      // first delete previous data
      const used_shapes = [];
      let max_entries = 1;

      for (let n = 0; n < draw_nodes.length; ++n) {
         const entry = draw_nodes[n];
         if (entry.done) continue;

         // shape can be provided with entry itself
         const shape = entry.server_shape || build_shapes[entry.shapeid];
         if (!shape || !shape.ready) {
            console.warn(`Problem with shape id ${entry.shapeid} when building`);
            return false;
         }

         // ignore shape without geometry
         if (!shape.geom || !shape.nfaces)
            continue;

         if (shape.instances === undefined) {
            shape.instances = [];
            used_shapes.push(shape);
         }

         const instance = shape.instances.find(i => i.nodeid === entry.nodeid);

         if (instance) {
            instance.entries.push(entry);
            max_entries = Math.max(max_entries, instance.entries.length);
         } else
            shape.instances.push({ nodeid: entry.nodeid, entries: [entry] });
      }

      const make_sense = ctrl.instancing > 0 ? (max_entries > 2) : (draw_nodes.length > 10000) && (max_entries > 10);

      if (!make_sense) {
         used_shapes.forEach(shape => { delete shape.instances; });
         return false;
      }

      used_shapes.forEach(shape => {
         shape.used = true;
         shape.instances.forEach(instance => {
            const entry0 = instance.entries[0],
                prop = this.getDrawEntryProperties(entry0, colors);

            prop.material.wireframe = ctrl.wireframe;

            prop.material.side = ctrl.doubleside ? THREE.DoubleSide : THREE.FrontSide;

            if (instance.entries.length === 1)
               this.createEntryMesh(ctrl, toplevel, entry0, shape, colors);
            else {
               const arr1 = [], arr2 = [], stacks1 = [], stacks2 = [], names1 = [], names2 = [];

               instance.entries.forEach(entry => {
                  const info = this.resolveStack(entry.stack, true);
                  if (info.matrix.determinant() > -0.9) {
                     arr1.push(info.matrix);
                     stacks1.push(entry.stack);
                     names1.push(this.getNodeName(entry.nodeid));
                  } else {
                     arr2.push(info.matrix);
                     stacks2.push(entry.stack);
                     names2.push(this.getNodeName(entry.nodeid));
                  }
                  entry.done = true;
               });

               if (arr1.length) {
                  const mesh1 = new THREE.InstancedMesh(shape.geom, prop.material, arr1.length);

                  mesh1.stacks = stacks1;
                  arr1.forEach((matrix, i) => mesh1.setMatrixAt(i, matrix));

                  toplevel.add(mesh1);

                  mesh1.renderOrder = 1;

                  if (ctrl.set_names) {
                     mesh1.name = names1[0];
                     mesh1.names = names1;
                  }

                  if (ctrl.set_origin)
                     mesh1.userData = prop.volume;

                  mesh1.$jsroot_order = 1;
                  ctrl.info.num_meshes++;
                  ctrl.info.num_faces += shape.nfaces * arr1.length;
               }

               if (arr2.length) {
                  if (shape.geomZ === undefined)
                     shape.geomZ = createFlippedGeom(shape.geom);

                  const mesh2 = new THREE.InstancedMesh(shape.geomZ, prop.material, arr2.length);

                  mesh2.stacks = stacks2;
                  const m = new THREE.Matrix4().makeScale(1, 1, -1);
                  arr2.forEach((matrix, i) => {
                     mesh2.setMatrixAt(i, matrix.multiply(m));
                  });
                  mesh2._flippedMesh = true;

                  toplevel.add(mesh2);

                  mesh2.renderOrder = 1;
                  if (ctrl.set_names) {
                     mesh2.name = names2[0];
                     mesh2.names = names2;
                  }
                  if (ctrl.set_origin)
                     mesh2.userData = prop.volume;

                  mesh2.$jsroot_order = 1;
                  ctrl.info.num_meshes++;
                  ctrl.info.num_faces += shape.nfaces*arr2.length;
               }
            }
         });

         delete shape.instances;
      });

      return true;
   }

   /** @summary Get volume boundary */
   getVolumeBoundary(viscnt, facelimit, nodeslimit) {
      const result = { min: 0, max: 1, sortidcut: 0 };

      if (!this.sortmap) {
         console.error('sorting map do not exist');
         return result;
      }

      let maxNode, currNode, cnt=0, facecnt=0;

      for (let n = 0; (n < this.sortmap.length) && (cnt < nodeslimit) && (facecnt < facelimit); ++n) {
         const id = this.sortmap[n];
         if (viscnt[id] === 0) continue;
         currNode = this.nodes[id];
         if (!maxNode) maxNode = currNode;
         cnt += viscnt[id];
         facecnt += viscnt[id] * currNode.nfaces;
      }

      if (!currNode) {
         console.error('no volumes selected');
         return result;
      }

      result.max = maxNode.vol;
      result.min = currNode.vol;
      result.sortidcut = currNode.sortid; // latest node is not included
      return result;
   }

   /** @summary Collects visible nodes, using maxlimit
     * @desc One can use map to define cut based on the volume or serious of cuts */
   collectVisibles(maxnumfaces, frustum) {
      // in simple case shape as it is
      if (this.plain_shape)
         return { lst: [{ nodeid: 0, seqid: 0, stack: [], factor: 1, shapeid: 0, server_shape: this.plain_shape }], complete: true };

      const arg = {
         facecnt: 0,
         viscnt: new Array(this.nodes.length), // counter for each node
         vislvl: this.getVisLevel(),
         reset() {
            this.total = 0;
            this.facecnt = 0;
            this.viscnt.fill(0);
         },
         func(node) {
            this.total++;
            this.facecnt += node.nfaces;
            this.viscnt[node.id]++;
            return true;
         }
      };

      arg.reset();

      let total = this.scanVisible(arg);
      if ((total === 0) && (this.nodes[0].vis < 2) && !this.nodes[0].nochlds) {
         // try to draw only main node by default
         arg.reset();
         arg.main_visible = true;
         total = this.scanVisible(arg);
      }

      const maxnumnodes = this.getMaxVisNodes();

      if (maxnumnodes > 0) {
         while ((total > maxnumnodes) && (arg.vislvl > 1)) {
            arg.vislvl--;
            arg.reset();
            total = this.scanVisible(arg);
         }
      }

      this.actual_level = arg.vislvl; // not used, can be shown somewhere in the gui

      let minVol = 0, maxVol, camVol = -1, camFact = 10, sortidcut = this.nodes.length + 1;

      if (arg.facecnt > maxnumfaces) {
         const bignumfaces = maxnumfaces * (frustum ? 0.8 : 1.0),
               bignumnodes = maxnumnodes * (frustum ? 0.8 : 1.0),
               // define minimal volume, which always to shown
               boundary = this.getVolumeBoundary(arg.viscnt, bignumfaces, bignumnodes);

         minVol = boundary.min;
         maxVol = boundary.max;
         sortidcut = boundary.sortidcut;

         if (frustum) {
             arg.domatrix = true;
             arg.frustum = frustum;
             arg.totalcam = 0;
             arg.func = function(node) {
                if (node.vol <= minVol) {
                    // only small volumes are interesting
                    if (this.frustum.CheckShape(this.getmatrix(), node)) {
                      this.viscnt[node.id]++;
                      this.totalcam += node.nfaces;
                   }
                }

                return true;
             };

            for (let n = 0; n < arg.viscnt.length; ++n)
               arg.viscnt[n] = 0;

             this.scanVisible(arg);

             if (arg.totalcam > maxnumfaces*0.2)
                camVol = this.getVolumeBoundary(arg.viscnt, maxnumfaces*0.2, maxnumnodes*0.2).min;
             else
                camVol = 0;

             camFact = maxVol / ((camVol > 0) ? (camVol > 0) : minVol);
         }
      }

      arg.items = [];

      arg.func = function(node) {
         if (node.sortid < sortidcut)
            this.items.push(this.copyStack());
          else if ((camVol >= 0) && (node.vol > camVol)) {
            if (this.frustum.CheckShape(this.getmatrix(), node))
               this.items.push(this.copyStack(camFact));
         }
         return true;
      };

      this.scanVisible(arg);

      return { lst: arg.items, complete: minVol === 0 };
   }

   /** @summary Merge list of drawn objects
     * @desc In current list we should mark if object already exists
     * from previous list we should collect objects which are not there */
   mergeVisibles(current, prev) {
      let indx2 = 0;
      const del = [];
      for (let indx1 = 0; (indx1 < current.length) && (indx2 < prev.length); ++indx1) {
         while ((indx2 < prev.length) && (prev[indx2].seqid < current[indx1].seqid))
            del.push(prev[indx2++]); // this entry should be removed


         if ((indx2 < prev.length) && (prev[indx2].seqid === current[indx1].seqid)) {
            if (prev[indx2].done) current[indx1].done = true; // copy ready flag
            indx2++;
         }
      }

      // remove rest
      while (indx2 < prev.length)
         del.push(prev[indx2++]);

      return del;
   }

   /** @summary Collect all uniques shapes which should be built
    *  @desc Check if same shape used many times for drawing */
   collectShapes(lst) {
      // nothing else - just that single shape
      if (this.plain_shape)
         return [this.plain_shape];

      const shapes = [];

      for (let i = 0; i < lst.length; ++i) {
         const entry = lst[i],
             shape = this.getNodeShape(entry.nodeid);

         if (!shape) continue; // strange, but avoid misleading

         if (shape._id === undefined) {
            shape._id = shapes.length;

            shapes.push({ id: shape._id, shape, vol: this.nodes[entry.nodeid].vol, refcnt: 1, factor: 1, ready: false });

            // shapes.push( { obj: shape, vol: this.nodes[entry.nodeid].vol });
         } else
            shapes[shape._id].refcnt++;


         entry.shape = shapes[shape._id]; // remember shape used

         // use maximal importance factor to push element to the front
         if (entry.factor && (entry.factor>entry.shape.factor))
            entry.shape.factor = entry.factor;
      }

      // now sort shapes in volume decrease order
      shapes.sort((a, b) => b.vol*b.factor - a.vol*a.factor);

      // now set new shape ids according to the sorted order and delete temporary field
      for (let n = 0; n < shapes.length; ++n) {
         const item = shapes[n];
         item.id = n; // set new ID
         delete item.shape._id; // remove temporary field
      }

      // as last action set current shape id to each entry
      for (let i = 0; i < lst.length; ++i) {
         const entry = lst[i];
         if (entry.shape) {
            entry.shapeid = entry.shape.id; // keep only id for the entry
            delete entry.shape; // remove direct references
         }
      }

      return shapes;
   }

   /** @summary Merge shape lists */
   mergeShapesLists(oldlst, newlst) {
      if (!oldlst) return newlst;

      // set geometry to shape object itself
      for (let n = 0; n < oldlst.length; ++n) {
         const item = oldlst[n];

         item.shape._geom = item.geom;
         delete item.geom;

         if (item.geomZ !== undefined) {
            item.shape._geomZ = item.geomZ;
            delete item.geomZ;
         }
      }

      // take from shape (if match)
      for (let n = 0; n < newlst.length; ++n) {
         const item = newlst[n];

         if (item.shape._geom !== undefined) {
            item.geom = item.shape._geom;
            delete item.shape._geom;
         }

         if (item.shape._geomZ !== undefined) {
            item.geomZ = item.shape._geomZ;
            delete item.shape._geomZ;
         }
      }

      // now delete all unused geometries
      for (let n = 0; n < oldlst.length; ++n) {
         const item = oldlst[n];
         delete item.shape._geom;
         delete item.shape._geomZ;
      }

      return newlst;
   }

   /** @summary Build shapes */
   buildShapes(lst, limit, timelimit) {
      let created = 0;
      const tm1 = new Date().getTime(),
            res = { done: false, shapes: 0, faces: 0, notusedshapes: 0 };

      for (let n = 0; n < lst.length; ++n) {
         const item = lst[n];

         // if enough faces are produced, nothing else is required
         if (res.done) { item.ready = true; continue; }

         if (!item.ready) {
            item._typename = '$$Shape$$'; // let reuse item for direct drawing
            item.ready = true;
            if (item.geom === undefined) {
               item.geom = createGeometry(item.shape);
               if (item.geom) created++; // indicate that at least one shape was created
            }
            item.nfaces = countGeometryFaces(item.geom);
         }

         res.shapes++;
         if (!item.used) res.notusedshapes++;
         res.faces += item.nfaces * item.refcnt;

         if (res.faces >= limit)
            res.done = true;
         else if ((created > 0.01*lst.length) && (timelimit !== undefined)) {
            const tm2 = new Date().getTime();
            if (tm2 - tm1 > timelimit) return res;
         }
      }

      res.done = true;

      return res;
   }

   /** @summary Format REveGeomNode data to be able use it in list of clones
     * @private */
   static formatServerElement(elem) {
      elem.kind = 2; // special element for geom viewer, used in TGeoPainter
      elem.vis = 2; // visibility is alwys on
      const m = elem.matr;
      delete elem.matr;
      if (!m?.length) return elem;

      if (m.length === 16)
         elem.matrix = m;
       else {
         const nm = elem.matrix = new Array(16);
         nm.fill(0);
         nm[0] = nm[5] = nm[10] = nm[15] = 1;

         if (m.length === 3) {
            // translation matrix
            nm[12] = m[0]; nm[13] = m[1]; nm[14] = m[2];
         } else if (m.length === 4) {
            // scale matrix
            nm[0] = m[0]; nm[5] = m[1]; nm[10] = m[2]; nm[15] = m[3];
         } else if (m.length === 9) {
            // rotation matrix
            nm[0] = m[0]; nm[4] = m[1]; nm[8] = m[2];
            nm[1] = m[3]; nm[5] = m[4]; nm[9] = m[5];
            nm[2] = m[6]; nm[6] = m[7]; nm[10] = m[8];
         } else
            console.error(`wrong number of elements ${m.length} in the matrix`);
      }
      return elem;
   }

} // class ClonedNodes

/** @summary extract code of Box3.expandByObject
  * @desc Major difference - do not traverse hierarchy, support InstancedMesh
  * @private */
function getBoundingBox(node, box3, local_coordinates) {
   if (!node?.geometry) return box3;

   if (!box3) box3 = new THREE.Box3().makeEmpty();

   if (node.isInstancedMesh) {
      const m = new THREE.Matrix4(), b = new THREE.Box3().makeEmpty();

      node.geometry.computeBoundingBox();

      for (let i = 0; i < node.count; i++) {
         node.getMatrixAt(i, m);
         b.copy(node.geometry.boundingBox).applyMatrix4(m);
         box3.union(b);
      }
      return box3;
   }

   if (!local_coordinates) node.updateWorldMatrix(false, false);

   const v1 = new THREE.Vector3(), attribute = node.geometry.attributes?.position;

   if (attribute !== undefined) {
      for (let i = 0, l = attribute.count; i < l; i++) {
         // v1.fromAttribute( attribute, i ).applyMatrix4( node.matrixWorld );
         v1.fromBufferAttribute(attribute, i);
         if (!local_coordinates) v1.applyMatrix4(node.matrixWorld);
         box3.expandByPoint(v1);
      }
   }

   return box3;
}

/** @summary Cleanup shape entity
  * @private */
function cleanupShape(shape) {
   if (!shape) return;

   if (isFunc(shape.geom?.dispose))
      shape.geom.dispose();

   if (isFunc(shape.geomZ?.dispose))
      shape.geomZ.dispose();

   delete shape.geom;
   delete shape.geomZ;
}

/** @summary Set rendering order for created hierarchy
  * @desc depending from provided method sort differently objects
  * @param toplevel - top element
  * @param origin - camera position used to provide sorting
  * @param method - name of sorting method like 'pnt', 'ray', 'size', 'dflt'  */
function produceRenderOrder(toplevel, origin, method, clones) {
   const raycast = new THREE.Raycaster();

   function setdefaults(top) {
      if (!top) return;
      top.traverse(obj => {
         obj.renderOrder = obj.defaultOrder || 0;
         if (obj.material) obj.material.depthWrite = true; // by default depthWriting enabled
      });
   }

   function traverse(obj, lvl, arr) {
      // traverse hierarchy and extract all children of given level
      // if (obj.$jsroot_depth === undefined) return;

      if (!obj.children) return;

      for (let k = 0; k < obj.children.length; ++k) {
         const chld = obj.children[k];
         if (chld.$jsroot_order === lvl) {
            if (chld.material) {
               if (chld.material.transparent) {
                  chld.material.depthWrite = false; // disable depth writing for transparent
                  arr.push(chld);
               } else
                  setdefaults(chld);
            }
         } else if ((obj.$jsroot_depth === undefined) || (obj.$jsroot_depth < lvl))
            traverse(chld, lvl, arr);
      }
   }

   function sort(arr, minorder, maxorder) {
      // resort meshes using ray caster and camera position
      // idea to identify meshes which are in front or behind

      if (arr.length > 1000) {
         // too many of them, just set basic level and exit
         for (let i = 0; i < arr.length; ++i)
            arr[i].renderOrder = (minorder + maxorder)/2;
         return false;
      }

      const tmp_vect = new THREE.Vector3();

      // first calculate distance to the camera
      // it gives preliminary order of volumes
      for (let i = 0; i < arr.length; ++i) {
         const mesh = arr[i];
         let box3 = mesh.$jsroot_box3;

         if (!box3)
            mesh.$jsroot_box3 = box3 = getBoundingBox(mesh);

         if (method === 'size') {
            const sz = box3.getSize(new THREE.Vector3());
            mesh.$jsroot_distance = sz.x*sz.y*sz.z;
            continue;
         }

         if (method === 'pnt') {
            mesh.$jsroot_distance = origin.distanceTo(box3.getCenter(tmp_vect));
            continue;
         }

         let dist = Math.min(origin.distanceTo(box3.min), origin.distanceTo(box3.max));
         const pnt = new THREE.Vector3(box3.min.x, box3.min.y, box3.max.z);

         dist = Math.min(dist, origin.distanceTo(pnt));
         pnt.set(box3.min.x, box3.max.y, box3.min.z);
         dist = Math.min(dist, origin.distanceTo(pnt));
         pnt.set(box3.max.x, box3.min.y, box3.min.z);
         dist = Math.min(dist, origin.distanceTo(pnt));
         pnt.set(box3.max.x, box3.max.y, box3.min.z);
         dist = Math.min(dist, origin.distanceTo(pnt));
         pnt.set(box3.max.x, box3.min.y, box3.max.z);
         dist = Math.min(dist, origin.distanceTo(pnt));
         pnt.set(box3.min.x, box3.max.y, box3.max.z);
         dist = Math.min(dist, origin.distanceTo(pnt));

         mesh.$jsroot_distance = dist;
      }

      arr.sort((a, b) => a.$jsroot_distance - b.$jsroot_distance);

      const resort = new Array(arr.length);

      for (let i = 0; i < arr.length; ++i) {
         arr[i].$jsroot_index = i;
         resort[i] = arr[i];
      }

      if (method === 'ray') {
         for (let i = arr.length - 1; i >= 0; --i) {
            const mesh = arr[i], box3 = mesh.$jsroot_box3;
            let intersects, direction = box3.getCenter(tmp_vect);

            for (let ntry = 0; ntry < 2; ++ntry) {
               direction.sub(origin).normalize();

               raycast.set(origin, direction);

               intersects = raycast.intersectObjects(arr, false) || []; // only plain array
               const unique = [];

               for (let k1 = 0; k1 < intersects.length; ++k1) {
                  if (unique.indexOf(intersects[k1].object) < 0)
                     unique.push(intersects[k1].object);
                  // if (intersects[k1].object === mesh) break; // trace until object itself
               }

               intersects = unique;

               if ((intersects.indexOf(mesh) < 0) && (ntry > 0))
                  console.log(`MISS ${clones?.resolveStack(mesh.stack)?.name}`);

               if ((intersects.indexOf(mesh) >= 0) || (ntry > 0)) break;

               const pos = mesh.geometry.attributes.position.array;

               direction = new THREE.Vector3((pos[0]+pos[3]+pos[6])/3, (pos[1]+pos[4]+pos[7])/3, (pos[2]+pos[5]+pos[8])/3);

               direction.applyMatrix4(mesh.matrixWorld);
            }

            // now push first object in intersects to the front
            for (let k1 = 0; k1 < intersects.length - 1; ++k1) {
               const mesh1 = intersects[k1], mesh2 = intersects[k1+1],
                     i1 = mesh1.$jsroot_index, i2 = mesh2.$jsroot_index;
               if (i1 < i2) continue;
               for (let ii = i2; ii < i1; ++ii) {
                  resort[ii] = resort[ii+1];
                  resort[ii].$jsroot_index = ii;
               }
               resort[i1] = mesh2;
               mesh2.$jsroot_index = i1;
            }
         }
      }

      for (let i = 0; i < resort.length; ++i) {
         resort[i].renderOrder = Math.round(maxorder - (i+1) / (resort.length + 1) * (maxorder - minorder));
         delete resort[i].$jsroot_index;
         delete resort[i].$jsroot_distance;
      }

      return true;
   }

   function process(obj, lvl, minorder, maxorder) {
      const arr = [];
      let did_sort = false;

      traverse(obj, lvl, arr);

      if (!arr.length) return;

      if (minorder === maxorder) {
         for (let k = 0; k < arr.length; ++k)
            arr[k].renderOrder = minorder;
      } else {
        did_sort = sort(arr, minorder, maxorder);
        if (!did_sort) minorder = maxorder = (minorder + maxorder) / 2;
      }

      for (let k = 0; k < arr.length; ++k) {
         const next = arr[k].parent;
         let min = minorder, max = maxorder;

         if (did_sort) {
            max = arr[k].renderOrder;
            min = max - (maxorder - minorder) / (arr.length + 2);
         }

         process(next, lvl+1, min, max);
      }
   }

   if (!method || (method === 'dflt'))
      setdefaults(toplevel);
   else
      process(toplevel, 0, 1, 1000000);
}

/** @summary provide icon name for the shape
  * @private */
function getShapeIcon(shape) {
   switch (shape._typename) {
      case clTGeoArb8: return 'img_geoarb8';
      case clTGeoCone: return 'img_geocone';
      case clTGeoConeSeg: return 'img_geoconeseg';
      case clTGeoCompositeShape: return 'img_geocomposite';
      case clTGeoTube: return 'img_geotube';
      case clTGeoTubeSeg: return 'img_geotubeseg';
      case clTGeoPara: return 'img_geopara';
      case clTGeoParaboloid: return 'img_geoparab';
      case clTGeoPcon: return 'img_geopcon';
      case clTGeoPgon: return 'img_geopgon';
      case clTGeoShapeAssembly: return 'img_geoassembly';
      case clTGeoSphere: return 'img_geosphere';
      case clTGeoTorus: return 'img_geotorus';
      case clTGeoTrd1: return 'img_geotrd1';
      case clTGeoTrd2: return 'img_geotrd2';
      case clTGeoXtru: return 'img_geoxtru';
      case clTGeoTrap: return 'img_geotrap';
      case clTGeoGtra: return 'img_geogtra';
      case clTGeoEltu: return 'img_geoeltu';
      case clTGeoHype: return 'img_geohype';
      case clTGeoCtub: return 'img_geoctub';
   }
   return 'img_geotube';
}

function runGeoWorker(ctxt, data, doPost) {
   if (isStr(data)) {
      console.log(`Worker get message ${data}`);
      return;
   }

   if (!isObject(data))
      return;

   data.tm1 = new Date().getTime();

   if (data.init) {
      // console.log(`start worker ${data.tm1 -  data.tm0}`);

      if (data.clones) {
         // console.log(`get clones ${nodes.length}`);
         ctxt.clones = new ClonedNodes(null, data.clones);
         ctxt.clones.setVisLevel(data.vislevel);
         ctxt.clones.setMaxVisNodes(data.maxvisnodes);
         ctxt.clones.sortmap = data.sortmap;
         delete data.clones;
      }

      data.tm2 = new Date().getTime();

      return doPost(data);
   }

   if (data.shapes) {
      // this is task to create geometries in the worker

      const shapes = data.shapes, transferables = [];

      // build all shapes up to specified limit, also limit execution time
      for (let n = 0; n < 100; ++n) {
         const res = ctxt.clones.buildShapes(shapes, data.limit, 1000);
         if (res.done) break;
         doPost({ progress: `Worker creating: ${res.shapes} / ${shapes.length} shapes, ${res.faces} faces` });
      }

      for (let n = 0; n < shapes.length; ++n) {
         const item = shapes[n];

         if (item.geom) {
            let bufgeom;
            if (item.geom instanceof THREE.BufferGeometry)
               bufgeom = item.geom;
            else
               bufgeom = new THREE.BufferGeometry().fromGeometry(item.geom);

            item.buf_pos = bufgeom.attributes.position.array;
            item.buf_norm = bufgeom.attributes.normal.array;

            // use nice feature of HTML workers with transferable
            // we allow to take ownership of buffer from local array
            // therefore buffer content not need to be copied
            transferables.push(item.buf_pos.buffer, item.buf_norm.buffer);

            delete item.geom;
         }

         delete item.shape; // no need to send back shape
      }

      data.tm2 = new Date().getTime();

      return doPost(data, transferables);
   }

   if (data.collect !== undefined) {
      // this is task to collect visible nodes using camera position

      // first mark all visible flags
      ctxt.clones.setVisibleFlags(data.flags);
      ctxt.clones.setVisLevel(data.vislevel);
      ctxt.clones.setMaxVisNodes(data.maxvisnodes);

      delete data.flags;

      ctxt.clones.produceIdShifts();

      let matrix = null;
      if (data.matrix)
         matrix = new THREE.Matrix4().fromArray(data.matrix);
      delete data.matrix;

      const res = ctxt.clones.collectVisibles(data.collect, createFrustum(matrix));

      data.new_nodes = res.lst;
      data.complete = res.complete; // inform if all nodes are selected

      data.tm2 = new Date().getTime();

      return doPost(data);
   }
}

export { kindGeo, kindEve, kindShape,
         clTGeoBBox, clTGeoCompositeShape,
         geoCfg, geoBITS, ClonedNodes, isSameStack, checkDuplicates, getObjectName, testGeoBit, setGeoBit, toggleGeoBit,
         setInvisibleAll, countNumShapes, getNodeKind, produceRenderOrder, createFlippedGeom, createFlippedMesh, cleanupShape,
         createGeometry, numGeometryFaces, numGeometryVertices, createServerGeometry, createMaterial,
         projectGeometry, countGeometryFaces, createFrustum, createProjectionMatrix,
         getBoundingBox, provideObjectInfo, getShapeIcon, runGeoWorker };
