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

import java.util.HashMap;
import java.util.Map;

/**
 * Ranking matrix factorization with diversity
 *
 * @param <U> type of the users
 * @param <I> type of the items
 * @author Jacek Wasilewski (jacek.wasilewski@insightcentre.org)
 */
public class DivRankALSFactorizer<U, I> extends ALSFactorizer<U, I> {

    private static final Algebra ALG = new Algebra();
    private final double lambdaD;
    private final Boolean supportBasedWeighting;
    private final boolean useSimilarity;
    private final ItemDistanceModel<I> distanceModel;
    private final DiversityRegulariser regulariser;

    private boolean pre = false;

    private DoubleMatrix1D s;
    private double sum_s;
    private DoubleMatrix1D sum_sq;
    private DoubleMatrix2D sum_sqq;
    private DoubleMatrix1D divDot;

    public DivRankALSFactorizer(double lambdaD, ItemDistanceModel<I> distanceModel,
                                int numIter) {
        this(lambdaD, distanceModel, numIter, false, DiversityRegulariser.NONE, false);
    }

    public DivRankALSFactorizer(double lambdaD, ItemDistanceModel<I> distanceModel,
                                int numIter, boolean supportBasedWeighting) {
        this(lambdaD, distanceModel, numIter, supportBasedWeighting, DiversityRegulariser.NONE, false);
    }

    public DivRankALSFactorizer(double lambdaD, ItemDistanceModel<I> distanceModel,
                                int numIter, boolean supportBasedWeighting, DiversityRegulariser regulariser) {
        this(lambdaD, distanceModel, numIter, supportBasedWeighting, regulariser, false);
    }

    public DivRankALSFactorizer(double lambdaD, ItemDistanceModel<I> distanceModel,
                                int numIter, boolean supportBasedWeighting, DiversityRegulariser regulariser, boolean useSimilarity) {
        super(numIter);
        this.lambdaD = lambdaD;
        this.distanceModel = distanceModel;
        this.supportBasedWeighting = supportBasedWeighting;
        this.regulariser = regulariser;
        this.useSimilarity = useSimilarity;
    }

    @Override
    public double error(DenseDoubleMatrix2D p, DenseDoubleMatrix2D q, FastPreferenceData<U, I> data) {
        throw new RuntimeException("Not implemented.");
    }

    @Override
    public void set_minP(final DenseDoubleMatrix2D p, final DenseDoubleMatrix2D q, FastPreferenceData<U, I> data) {
        final int K = p.columns();
        final int nusers = data.numUsers();

        if (!pre) {
            pre(p, q, data);
            pre = true;
        }

        sum_sq = new DenseDoubleMatrix1D(K);
        sum_sqq = new DenseDoubleMatrix2D(K, K);
        data.getAllIidx().forEach(jidx -> {
            DoubleMatrix1D qj = q.viewRow(jidx);
            double sj = s.get(jidx);
            sum_sq.assign(qj, (x, y) -> x + y * sj);
            sum_sqq.assign(ALG.multOuter(qj, qj, null), (x, y) -> x + y * sj);
        });

        DoubleMatrix2D QtdivQ = new DenseDoubleMatrix2D(K, K);
        if (regulariser == DiversityRegulariser.P_LAPLACIAN_DQ) {
            DoubleMatrix1D[] sum_d_q = new DenseDoubleMatrix1D[data.numItems()];
            DoubleMatrix2D[] Ls = new DenseDoubleMatrix2D[data.numItems()];
            data.getAllItems().parallel().forEach(i -> {
                int iidx = data.item2iidx(i);
                sum_d_q[iidx] = new DenseDoubleMatrix1D(K);
                data.getAllItems().forEach(j -> {
                    int jidx = data.item2iidx(j);
                    sum_d_q[iidx].assign(q.viewRow(jidx), (a, b) -> a + (useSimilarity ? (1 - distanceModel.dist(i, j)) : distanceModel.dist(i, j)) * b);
                });
                DoubleMatrix1D L = new DenseDoubleMatrix1D(K);
                DoubleMatrix1D qi = q.viewRow(iidx);
                L.assign(sum_d_q[iidx]);
                L.assign(qi, (x, y) -> x - y * divDot.getQuick(iidx));
                Ls[iidx] = ALG.multOuter(qi, L, null);
            });

            data.getAllIidx().forEach(iidx -> {
                QtdivQ.assign(Ls[iidx], (x, y) -> x + y);
            });
        }

        data.getUidxWithPreferences().parallel().forEach(uidx -> {
            DoubleMatrix2D sum_cqq = new DenseDoubleMatrix2D(K, K);
            DoubleMatrix1D sum_cq = new DenseDoubleMatrix1D(K);
            DoubleMatrix1D sum_cqr = new DenseDoubleMatrix1D(K);
            DoubleMatrix1D sum_sqr = new DenseDoubleMatrix1D(K);

            double sum_c = data.numItems(uidx);
            double[] sum_sr = new double[]{0};
            double[] sum_cr = new double[]{0};

            data.getUidxPreferences(uidx).forEach(iv -> {
                int iidx = iv.idx;
                double rui = iv.v;
                DoubleMatrix1D qi = q.viewRow(iidx);

                DoubleMatrix2D tmp = new DenseDoubleMatrix2D(K, K);
                ALG.multOuter(qi, qi, tmp);
                sum_cqq.assign(tmp, (x, y) -> x + y);
                sum_cq.assign(qi, (x, y) -> x + y);
                sum_cqr.assign(qi, (x, y) -> x + y * rui);

                double si = s.get(iidx);
                sum_sr[0] += si * rui;
                sum_cr[0] += rui;
                sum_sqr.assign(qi, (x, y) -> x + y * si * rui);
            });

            DoubleMatrix2D A = new DenseDoubleMatrix2D(K, K);
            DoubleMatrix1D b = new DenseDoubleMatrix1D(K);
            A.assign(sum_cqq, (x, y) -> x + y * sum_s);
            A.assign(ALG.multOuter(sum_cq, sum_sq, null), (x, y) -> x - y);
            A.assign(ALG.multOuter(sum_sq, sum_cq, null), (x, y) -> x - y);
            A.assign(sum_sqq, (x, y) -> x + y * sum_c);

            b.assign(sum_cqr, (x, y) -> x + y * sum_s);
            b.assign(sum_cq, (x, y) -> x - y * sum_sr[0]);
            b.assign(sum_sq, (x, y) -> x - y * sum_cr[0]);
            b.assign(sum_sqr, (x, y) -> x + y * sum_c);

            if (regulariser == DiversityRegulariser.P_LAPLACIAN_DQ) {
                DoubleMatrix1D divReg = new DenseDoubleMatrix1D(K);
                DoubleMatrix1D pu = p.viewRow(uidx);
                divReg.assign(ALG.mult(QtdivQ, pu), (x, y) -> x - y);
                b.assign(divReg, (x, y) -> x + lambdaD / nusers * y);
            }

            LUDecompositionQuick lu = new LUDecompositionQuick(0);
            lu.decompose(A);
            lu.solve(b);
            p.viewRow(uidx).assign(b);
        });
    }

    @Override
    public void set_minQ(final DenseDoubleMatrix2D q, final DenseDoubleMatrix2D p, FastPreferenceData<U, I> data) {
        final int K = q.columns();
        final int nusers = data.numUsers();

        if (!pre) {
            pre(p, q, data);
            pre = true;
        }

        sum_sq = new DenseDoubleMatrix1D(K);
        sum_sqq = new DenseDoubleMatrix2D(K, K);
        data.getAllIidx().forEach(jidx -> {
            DoubleMatrix1D qj = q.viewRow(jidx);
            double sj = s.get(jidx);
            sum_sq.assign(qj, (x, y) -> x + y * sj);
            sum_sqq.assign(ALG.multOuter(qj, qj, null), (x, y) -> x + y * sj);
        });

        Map<Integer, double[]> m_sum_sr = new HashMap<>();
        Map<Integer, double[]> m_sum_cr = new HashMap<>();
        Map<Integer, Double> m_sum_c = new HashMap<>();
        Map<Integer, DoubleMatrix1D> m_sum_cq = new HashMap<>();

        DoubleMatrix2D sum_p_p_c = new DenseDoubleMatrix2D(K, K);
        DoubleMatrix1D sum_p_p_cq = new DenseDoubleMatrix1D(K);
        DoubleMatrix1D sum_cr_p = new DenseDoubleMatrix1D(K);

        data.getUidxWithPreferences().forEach(uidx -> {
            double[] sum_sr = new double[]{0};
            double[] sum_cr = new double[]{0};
            double sum_c = data.numItems(uidx);
            DoubleMatrix1D sum_cq = new DenseDoubleMatrix1D(K);

            data.getUidxPreferences(uidx).forEach(jv -> {
                int jidx = jv.idx;
                double ruj = jv.v;
                double sj = s.get(jidx);

                sum_sr[0] += sj * ruj;
                sum_cr[0] += ruj;
                sum_cq.assign(q.viewRow(jidx), (x, y) -> x + y);
            });

            m_sum_sr.put(uidx, sum_sr);
            m_sum_cr.put(uidx, sum_cr);
            m_sum_c.put(uidx, sum_c);
            m_sum_cq.put(uidx, sum_cq);

            DoubleMatrix1D pu = p.viewRow(uidx);

            DoubleMatrix2D pp = new DenseDoubleMatrix2D(K, K);
            ALG.multOuter(pu, pu, pp);
            sum_p_p_c.assign(pp, (x, y) -> x + y * m_sum_c.get(uidx));
            DenseDoubleMatrix1D tmp = new DenseDoubleMatrix1D(K);
            pp.zMult(m_sum_cq.get(uidx), tmp);
            sum_p_p_cq.assign(tmp, (x, y) -> x + y);
            sum_cr_p.assign(pu, (x, y) -> x + y * m_sum_cr.get(uidx)[0]);
        });

        DoubleMatrix1D[] sum_d_q = new DenseDoubleMatrix1D[data.numItems()];
        if (regulariser != DiversityRegulariser.NONE) {
            data.getAllItems().parallel().forEach(i -> {
                int iidx = data.item2iidx(i);
                sum_d_q[iidx] = new DenseDoubleMatrix1D(K);
                data.getAllItems().forEach(j -> {
                    int jidx = data.item2iidx(j);
                    sum_d_q[iidx].assign(q.viewRow(jidx), (a, b) -> a + (useSimilarity ? (1 - distanceModel.dist(i, j)) : distanceModel.dist(i, j)) * b);
                });
            });
        }

        DoubleMatrix2D PtP = new DenseDoubleMatrix2D(K, K);
        if (regulariser == DiversityRegulariser.P_LAPLACIAN_DQ) {
            p.zMult(p, PtP, 1.0, 0.0, true, false);
        }

        data.getIidxWithPreferences().parallel().forEach(iidx -> {
            DoubleMatrix2D sum_cpp = new DenseDoubleMatrix2D(K, K);
            DoubleMatrix1D sum_cpr = new DenseDoubleMatrix1D(K);
            DoubleMatrix1D sum_c_sr_p = new DenseDoubleMatrix1D(K);
            DoubleMatrix1D sum_p_r_c = new DenseDoubleMatrix1D(K);

            double si = s.get(iidx);
            data.getIidxPreferences(iidx).forEach(uv -> {
                int uidx = uv.idx;
                DoubleMatrix1D pu = p.viewRow(uidx);
                double rui = uv.v;

                DoubleMatrix2D pp = new DenseDoubleMatrix2D(K, K);
                ALG.multOuter(pu, pu, pp);

                sum_cpp.assign(pp, (x, y) -> x + y);


                sum_cpr.assign(pu, (x, y) -> x + y * rui);
                sum_c_sr_p.assign(pu, (x, y) -> x + y * m_sum_sr.get(uidx)[0]);
                sum_p_r_c.assign(pu, (x, y) -> x + y * rui * m_sum_c.get(uidx));
            });

            DoubleMatrix2D A = new DenseDoubleMatrix2D(K, K);
            DoubleMatrix1D b = new DenseDoubleMatrix1D(K);
            A.assign(sum_cpp, (x, y) -> x + y * sum_s);
            A.assign(sum_p_p_c, (x, y) -> x + y * si);

            b.assign(ALG.mult(sum_cpp, sum_sq), (x, y) -> x + y);
            b.assign(sum_cpr, (x, y) -> x + y * sum_s);
            b.assign(sum_c_sr_p, (x, y) -> x - y);
            b.assign(sum_p_p_cq, (x, y) -> x + y * si);
            b.assign(sum_cr_p, (x, y) -> x - y * si);
            b.assign(sum_p_r_c, (x, y) -> x + y * si);

            switch (regulariser) {
                case LAPLACIAN_DQ: {
                    DoubleMatrix1D divReg = new DenseDoubleMatrix1D(K);
                    DoubleMatrix1D qi = q.viewRow(iidx);
                    divReg.assign(sum_d_q[iidx]);
                    divReg.assign(qi, (x, y) -> x - y * divDot.getQuick(iidx));
                    b.assign(divReg, (x, y) -> x + lambdaD * y);
                }
                break;
                case DQ: {
                    DoubleMatrix1D divReg = new DenseDoubleMatrix1D(K);
                    divReg.assign(sum_d_q[iidx], (x, y) -> x - y);
                    b.assign(divReg, (x, y) -> x + lambdaD * y);
                }
                break;
                case P_LAPLACIAN_DQ: {
                    DoubleMatrix1D divReg = new DenseDoubleMatrix1D(K);
                    DoubleMatrix1D qi = q.viewRow(iidx);
                    divReg.assign(sum_d_q[iidx]);
                    divReg.assign(qi, (x, y) -> x - y * divDot.getQuick(iidx));
                    divReg = ALG.mult(PtP, divReg);
                    b.assign(divReg, (x, y) -> x + lambdaD / nusers * y);
                }
                break;
            }

            LUDecompositionQuick lu = new LUDecompositionQuick(0);
            lu.decompose(A);
            lu.solve(b);
            q.viewRow(iidx).assign(b);
        });
    }

    private void pre(final DenseDoubleMatrix2D p, final DenseDoubleMatrix2D q, FastPreferenceData<U, I> data) {
        s = new DenseDoubleMatrix1D(data.numItems());
        sum_s = 0.0;
        data.getAllIidx().forEach(iidx -> {
            double si = supportBasedWeighting ? data.numUsers(iidx) : 1;
            s.setQuick(iidx, si);
            sum_s += si;
        });
        if (regulariser == DiversityRegulariser.LAPLACIAN_DQ || regulariser == DiversityRegulariser.P_LAPLACIAN_DQ) {
            divDot = new DenseDoubleMatrix1D(data.numItems());
            data.getAllItems().parallel().forEach(i -> {
                int iidx = data.item2iidx(i);
                divDot.setQuick(iidx, data.getAllItems().mapToDouble(j -> (useSimilarity ? 1.0 - distanceModel.dist(i, j) : distanceModel.dist(i, j))).sum());
            });
        }
    }

    public enum DiversityRegulariser {
        LAPLACIAN_DQ,
        DQ,
        P_LAPLACIAN_DQ,
        NONE
    }
}
