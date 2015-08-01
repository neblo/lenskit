/*
 * LensKit, an open source recommender systems toolkit.
 * Copyright 2010-2014 LensKit Contributors.  See CONTRIBUTORS.md.
 * Work on LensKit has been funded by the National Science Foundation under
 * grants IIS 05-34939, 08-08692, 08-12148, and 10-17697.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 51
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */
package org.grouplens.lenskit.mf.svdpp;

import it.unimi.dsi.fastutil.longs.LongSet;
import it.unimi.dsi.fastutil.longs.LongSortedSet;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.baseline.BaselineScorer;
import org.grouplens.lenskit.basic.AbstractItemScorer;
import org.grouplens.lenskit.collections.LongUtils;
import org.grouplens.lenskit.data.dao.UserEventDAO;
import org.grouplens.lenskit.data.event.Rating;
import org.grouplens.lenskit.data.event.Ratings;
import org.grouplens.lenskit.data.history.History;
import org.grouplens.lenskit.data.history.UserHistory;
import org.grouplens.lenskit.data.pref.PreferenceDomain;
import org.grouplens.lenskit.iterative.TrainingLoopController;
import org.grouplens.lenskit.mf.svd.BiasedMFKernel;
import org.grouplens.lenskit.mf.svd.DomainClampingKernel;
import org.grouplens.lenskit.mf.svd.DotProductKernel;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import org.grouplens.lenskit.vectors.VectorEntry;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import javax.inject.Inject;
import java.util.Vector;

/**
 * Do recommendations and predictions based on SVD matrix factorization.
 *
 * Recommendation is done based on folding-in.  The strategy is do a fold-in
 * operation as described in
 * <a href="http://www.grouplens.org/node/212">Sarwar et al., 2002</a> with the
 * user's ratings.
 *
 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
 */
public class SVDppItemScorer extends AbstractItemScorer {

    protected final SVDppModel model;
    protected final BiasedMFKernel kernel;
    private UserEventDAO dao;
    private final ItemScorer baselineScorer;
    private final int featureCount;

    @Nullable
    private final SVDppUpdateRule rule;

    /**
     * Construct the item scorer.
     *
     * @param dao      The DAO.
     * @param model    The model.
     * @param baseline The baseline scorer.  Be very careful when configuring a different baseline
     *                 at runtime than at model-build time; such a configuration is unlikely to
     *                 perform well.
     * @param rule     The update rule, or {@code null} (the default) to only use the user features
     *                 from the model. If provided, this update rule is used to update a user's
     *                 feature values based on their profile when scores are requested.
     */
    @Inject
    public SVDppItemScorer(UserEventDAO dao, SVDppModel model,
                           @BaselineScorer ItemScorer baseline,
                           @Nullable PreferenceDomain dom,
                           @Nullable @RuntimeUpdate SVDppUpdateRule rule) {
        // FIXME Unify requirement on update rule and DAO
        this.dao = dao;
        this.model = model;
        baselineScorer = baseline;
        this.rule = rule;

        if (dom == null) {
            kernel = new DotProductKernel();
        } else {
            kernel = new DomainClampingKernel(dom);
        }

        featureCount = model.getFeatureCount();
    }

    @Nullable
    public SVDppUpdateRule getUpdateRule() {
        return rule;
    }

    /**
     * Predict for a user using their preference array and history vector.
     *
     * @param user   The user's ID
     * @param uprefs The user's preference vector.
     * @param output The output vector, whose key domain is the items to predict for. It must
     *               be initialized to the user's baseline predictions.
     */
    private void computeScores(long user, RealVector uprefs, MutableSparseVector output) {
        for (VectorEntry e : output) {
            final long item = e.getKey();
            RealVector ivec = model.getItemVector(item);
            if (ivec == null) {
                // no item-vector, cannot make an informed prediction.
                // unset the baseline to note that we are not predicting for this item.
                output.unset(e);
            } else {
                double score = kernel.apply(e.getValue(), uprefs, ivec);
                output.set(e, score);
            }
        }
    }

    /**
     * Get estimates for all a user's ratings and the target items.
     *
     * @param user    The user ID.
     * @param ratings The user's ratings.
     * @param items   The target items.
     * @return Baseline predictions for all items either in the target set or the set of
     *         rated items.
     */
    private MutableSparseVector initialEstimates(long user, SparseVector ratings, LongSortedSet items) {
        LongSet allItems = LongUtils.setUnion(items, ratings.keySet());
        MutableSparseVector estimates = MutableSparseVector.create(allItems);
        baselineScorer.score(user, estimates);
        return estimates;
    }

    @Override
    public void score(long user, @Nonnull MutableSparseVector scores) {
        UserHistory<Rating> history = dao.getEventsForUser(user, Rating.class);
        if (history == null) {
            history = History.forUser(user);
        }
        SparseVector ratings = Ratings.userRatingVector(history);
        
        RealVector userFV = model.getUserVector(user);

        MutableSparseVector estimates = initialEstimates(user, ratings, scores.keyDomain());
        // propagate estimates to the output scores
        scores.set(estimates);

        // learning rates // TODO make customizable
        double learn_rate = 0.007;
        double reg_term = 0.015;

        if (!ratings.isEmpty() && rule != null) {

            for (VectorEntry r : ratings ){

                RealVector itemFV = model.getItemVector(r.getKey()); // TODO r.getKey() gets item's key right (not position in vector)?
                RealVector implicitFV = MatrixUtils.createRealVector(new double[featureCount]);

                // Calculate user-item profile ->  Qi o (Pu + |N(u)|^-1/2 * SUM Yj)
                for(VectorEntry ur : ratings){
                    implicitFV.combineToSelf(1, 1, model.getImplicitFeedbackVector(ur.getKey()));
                }
                implicitFV.mapMultiplyToSelf(Math.pow(ratings.size(), -0.5));
                implicitFV.add(userFV);  // user item profile - (Pu + |N(u)|^-1/2 * SUM Yj)
                assert itemFV != null;
                double user_item_profile = itemFV.dotProduct(implicitFV);

                double estimate = estimates.get(r.getKey());  // base score estimate
                double pred = estimate + user_item_profile;   // Calculate Prediction
                double rating_value = r.getValue();           // Actual Rating Value
                double error = rating_value - pred;           // Calculate Error

                // compute delta in user vector : Pu <- Pu + learn_rate * (error * Qi - reg_term * Pu)
                RealVector uvec_deltas;
                uvec_deltas = itemFV.combine(error, -(reg_term), userFV);
                uvec_deltas.mapMultiplyToSelf(learn_rate);

                // compute delta in item vector : Qi <- Qi + learn_rate * (error * (Pu + |N(u)|^-1/2 * SUM Yj) - reg_term * Qi)
                RealVector ivec_deltas;
                ivec_deltas = implicitFV.combine(error, -(reg_term), itemFV);
                ivec_deltas.mapMultiplyToSelf(learn_rate);

                // compute/set new implicit feedback vectors : Yj <- Yj + learn_rate * ( error * |N(u)|^-1/2 * Qi - reg_term * Yj)
                for(VectorEntry ur : ratings){
                    long item_index = ur.getKey();
                    RealVector temp_vec = itemFV.combine(error * Math.pow(ratings.size(), -0.5), -(reg_term), model.getImplicitFeedbackVector(item_index));
                    temp_vec.mapMultiplyToSelf(learn_rate);
                    temp_vec.combineToSelf(1, 1, model.getImplicitFeedbackVector(item_index));
                    model.setImplicitFeedbackVector(item_index, temp_vec);
                }

                // apply deltas
                model.setUserVector(user, uvec_deltas.combineToSelf(1, 1, userFV));
                model.setItemVector(r.getKey(), ivec_deltas.combineToSelf(1, 1, itemFV));
            }
//            for (int f = 0; f < featureCount; f++) {
//                trainUserFeature(user, uprefs, ratings, estimates, f); // changed updated to uprefs
//            }
//            //uprefs = updated;
        }
        // scores are the estimates, uprefs are trained up.
        //computeScores(user, uprefs, scores);
    }

//
//    private void trainUserFeature(long user, RealVector uprefs, SparseVector ratings,
//                                  MutableSparseVector estimates, int feature) {
//        assert rule != null;
//        assert uprefs.getDimension() == featureCount;
//        assert feature >= 0 && feature < featureCount;
//
//        int tailStart = feature + 1;
//        int tailSize = featureCount - feature - 1;
//        RealVector utail = uprefs.getSubVector(tailStart, tailSize);
//        MutableSparseVector tails = MutableSparseVector.create(ratings.keySet());
//        for (VectorEntry e: tails.view(VectorEntry.State.EITHER)) {
//            RealVector ivec = model.getItemVector(e.getKey());
//            if (ivec == null) {
//                // FIXME Do this properly
//                tails.set(e, 0);
//            } else {
//                ivec = ivec.getSubVector(tailStart, tailSize);
//                tails.set(e, utail.dotProduct(ivec));
//            }
//        }
//
//        double rmse = Double.MAX_VALUE;
//        TrainingLoopController controller = rule.getTrainingLoopController();
//        while (controller.keepTraining(rmse)) {
//            rmse = doFeatureIteration(user, uprefs, ratings, estimates, feature, tails);
//        }
//    }
//
//    private double doFeatureIteration(long user, RealVector uprefs,
//                                      SparseVector ratings, MutableSparseVector estimates,
//                                      int feature, SparseVector itemTails) {
//        assert rule != null;
//
//        SVDppUpdater updater = rule.createUpdater();
//        for (VectorEntry e: ratings) {
//            final long iid = e.getKey();
//            final RealVector ivec = model.getItemVector(iid);
//            final RealVector nvec = model.getImplicitFeedbackVector(user);
//            if (ivec == null) {
//                continue;
//            }
//
//            updater.prepare(feature, e.getValue(), estimates.get(iid),
//                            uprefs.getEntry(feature), ivec.getEntry(feature), nvec, itemTails.get(iid));
//            // Step 4: update user preferences
//            uprefs.addToEntry(feature, updater.getUserFeatureUpdate());
//        }
//        return updater.getRMSE();
//    }
}
