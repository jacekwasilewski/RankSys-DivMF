/*
 * Copyright (C) 2016 Insight Centre for Data Analytics http://insight-centre.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.insightcentre.ranksys.divmf;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import cern.colt.matrix.linalg.LUDecompositionQuick;
import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import es.uam.eps.ir.ranksys.mf.als.ALSFactorizer;
import es.uam.eps.ir.ranksys.novdiv.distance.ItemDistanceModel;

import java.util.function.DoubleUnaryOperator;

/**
 * Implicit matrix factorization of Hu, Koren and Volinsky with diversity.
 *
 * @param <U> type of the users
 * @param <I> type of the items
 * @author Jacek Wasilewski (jacek.wasilewski@insightcentre.org)
 */
public class DivHKVFactorizer<U, I> extends ALSFactorizer<U, I> {

    private static final Algebra ALG = new Algebra();
    private final double lambdaP;
    private final double lambdaQ;
    private final double lambdaD;
    private final ItemDistanceModel<I> distanceModel;
    private final DoubleUnaryOperator confidence;
    private boolean preQ = false;
    private DoubleMatrix1D divDot;
    private final boolean useSimilarity;

    public DivHKVFactorizer(double lambda, double lambdaD, ItemDistanceModel<I> distanceModel, DoubleUnaryOperator confidence, int numIter) {
        this(lambda, lambda, lambdaD, distanceModel, confidence, numIter, false);
    }

    public DivHKVFactorizer(double lambdaP, double lambdaQ, double lambdaD, ItemDistanceModel<I> distanceModel,
                            DoubleUnaryOperator confidence, int numIter, boolean useSimilarity) {
        super(numIter);
        this.lambdaP = lambdaP;
        this.lambdaQ = lambdaQ;
        this.lambdaD = lambdaD;
        this.distanceModel = distanceModel;
        this.confidence = confidence;
        this.useSimilarity = useSimilarity;
    }

    @Override
    public double error(DenseDoubleMatrix2D p, DenseDoubleMatrix2D q, FastPreferenceData<U, I> data) {
        throw new RuntimeException("Not implemented.");
    }

    @Override
    public void set_minP(final DenseDoubleMatrix2D p, final DenseDoubleMatrix2D q, FastPreferenceData<U, I> data) {
        final int K = p.columns();

        DenseDoubleMatrix2D A1P = new DenseDoubleMatrix2D(K, K);
        q.zMult(q, A1P, 1.0, 0.0, true, false);
        for (int k = 0; k < K; k++) {
            A1P.setQuick(k, k, lambdaP + A1P.getQuick(k, k));
        }

        DenseDoubleMatrix2D[] A2P = new DenseDoubleMatrix2D[q.rows()];
        data.getIidxWithPreferences().parallel().forEach(iidx -> {
            A2P[iidx] = new DenseDoubleMatrix2D(K, K);
            DoubleMatrix1D qi = q.viewRow(iidx);
            ALG.multOuter(qi, qi, A2P[iidx]);
        });

        data.getUidxWithPreferences().parallel().forEach(uidx -> {
            DoubleMatrix2D A = new DenseDoubleMatrix2D(K, K);
            DoubleMatrix1D b = new DenseDoubleMatrix1D(K);
            A.assign(A1P);
            b.assign(0.0);

            data.getUidxPreferences(uidx).forEach(iv -> {
                int iidx = iv.idx;
                double rui = iv.v;
                double cui = confidence.applyAsDouble(rui);

                DoubleMatrix1D qi = q.viewRow(iidx);

                A.assign(A2P[iidx], (x, y) -> x + y * (cui - 1.0));
                b.assign(qi, (x, y) -> x + y * rui * cui);
            });
            LUDecompositionQuick lu = new LUDecompositionQuick(0);
            lu.decompose(A);
            lu.solve(b);
            p.viewRow(uidx).assign(b);
        });
    }

    @Override
    public void set_minQ(final DenseDoubleMatrix2D q, final DenseDoubleMatrix2D p, FastPreferenceData<U, I> data) {
        final int K = q.columns();

        if (!preQ) {
            divDot = new DenseDoubleMatrix1D(data.numItems());
            data.getAllItems().parallel().forEach(i -> {
                int iidx = data.item2iidx(i);
                divDot.setQuick(iidx, data.getAllItems().mapToDouble(j -> (useSimilarity ? 1.0 - distanceModel.dist(i, j) : distanceModel.dist(i, j))).sum());
            });
            preQ = true;
        }

        DenseDoubleMatrix2D A1Q = new DenseDoubleMatrix2D(K, K);
        p.zMult(p, A1Q, 1.0, 0.0, true, false);
        for (int k = 0; k < K; k++) {
            A1Q.setQuick(k, k, lambdaQ + A1Q.getQuick(k, k));
        }

        DenseDoubleMatrix2D[] A2Q = new DenseDoubleMatrix2D[p.rows()];
        data.getUidxWithPreferences().parallel().forEach(uidx -> {
            A2Q[uidx] = new DenseDoubleMatrix2D(K, K);
            DoubleMatrix1D pu = p.viewRow(uidx);
            ALG.multOuter(pu, pu, A2Q[uidx]);
        });

        DoubleMatrix1D[] divQ = new DenseDoubleMatrix1D[data.numItems()];
        data.getAllItems().parallel().forEach(i -> {
            int iidx = data.item2iidx(i);
            divQ[iidx] = new DenseDoubleMatrix1D(K);
            data.getAllItems().forEach(j -> {
                int jidx = data.item2iidx(j);
                divQ[iidx].assign(q.viewRow(jidx), (a, b) -> a + (useSimilarity ? (1 - distanceModel.dist(i, j)) : distanceModel.dist(i, j)) * b);
            });
        });

        data.getIidxWithPreferences().parallel().forEach(iidx -> {
            DoubleMatrix2D A = new DenseDoubleMatrix2D(K, K);
            DoubleMatrix1D b = new DenseDoubleMatrix1D(K);
            A.assign(A1Q);
            b.assign(0.0);

            data.getIidxPreferences(iidx).forEach(uv -> {
                int uidx = uv.idx;
                double rui = uv.v;
                double cui = confidence.applyAsDouble(rui);

                DoubleMatrix1D pu = p.viewRow(uidx);

                A.assign(A2Q[uidx], (x, y) -> x + y * (cui - 1.0));
                b.assign(pu, (x, y) -> x + y * rui * cui);
            });

            DoubleMatrix1D qi = q.viewRow(iidx);
            DoubleMatrix1D divReg = new DenseDoubleMatrix1D(K);
            divReg.assign(divQ[iidx]);
            divReg.assign(qi, (x, y) -> x - y * divDot.getQuick(iidx));
            b.assign(divReg, (x, y) -> x + lambdaD * y);

//            for (int k = 0; k < K; k++) {
//                A.setQuick(k, k, A.getQuick(k, k) + lambdaD * divDot.getQuick(iidx));
//            }
//            b.assign(divQ[iidx], (x, y) -> x +  lambdaD * y);

            LUDecompositionQuick lu = new LUDecompositionQuick(0);
            lu.decompose(A);
            lu.solve(b);
            q.viewRow(iidx).assign(b);
        });
    }
}