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

import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.lang3.time.StopWatch;
import org.grouplens.lenskit.core.Transient;
import org.grouplens.lenskit.data.pref.IndexedPreference;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.iterative.TrainingLoopController;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import javax.inject.Provider;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;

/**
 * SVD recommender builder using gradient descent.
 *
 * // TODO add documentation
 *
 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
 */
public class SVDppModelBuilder implements Provider<SVDppModel> {
    private static Logger logger = LoggerFactory.getLogger(SVDppModelBuilder.class);

    protected final int featureCount;
    protected final PreferenceSnapshot snapshot;
    protected final double initialValue;

    protected final SVDppUpdateRule rule;

    @Inject
    public SVDppModelBuilder(@Transient @Nonnull PreferenceSnapshot snapshot,
                             @Transient @Nonnull SVDppUpdateRule rule,
                             @FeatureCount int featureCount,
                             @InitialFeatureValue double initVal) {
        this.featureCount = featureCount;
        this.initialValue = initVal;
        this.snapshot = snapshot;
        this.rule = rule;
    }


    @Override
    public SVDppModel get() {
        // setup userFeatures ( set random value from 0.00001 to ~0.1 )
        Random rand = new Random();
        int userCount = snapshot.getUserIds().size();
        RealMatrix userFeatures = MatrixUtils.createRealMatrix(userCount, featureCount);
        for (int i = 0; i < userCount; i++) {
            for (int j = 0; j < featureCount; j++) {
                userFeatures.setEntry(i, j, rand.nextDouble() * .1 + 0.00001);
            }
        }

        // setup itemFeatures ( set random value from 0.00001 to ~0.1 )
        int itemCount = snapshot.getItemIds().size();
        RealMatrix itemFeatures = MatrixUtils.createRealMatrix(itemCount, featureCount);
        for (int i = 0; i < itemCount; i++) {
            for (int j = 0; j < featureCount; j++) {
                itemFeatures.setEntry(i, j, rand.nextDouble() * .1 + 0.00001);
            }
        }

        // setup implicitFeatures ( set random value from 0.00001 to ~0.1 )
        RealMatrix implicitFeatures = MatrixUtils.createRealMatrix(itemCount, featureCount);
        for (int i = 0; i < itemCount; i++) {
            for (int j = 0; j < featureCount; j++) {
                implicitFeatures.setEntry(i, j, rand.nextDouble() * .1 + 0.00001);
            }
        }

        logger.debug("Learning rate is {}", rule.getLearningRate());
        logger.debug("Regularization term is {}", rule.getTrainingRegularization());

        logger.info("Building SVD with {} features for {} ratings",
                    featureCount, snapshot.getRatings().size());

        TrainingEstimator estimates = rule.makeEstimator(snapshot);

        List<FeatureInfo> featureInfo = new ArrayList<FeatureInfo>(featureCount);

        // get ratings
        Collection<IndexedPreference> ratings = snapshot.getRatings();

        int MAX_ITERATIONS = 30;

        for (int i = 0; i < MAX_ITERATIONS; i++){
            // train model (one epoch)
            for (IndexedPreference r : ratings) {
                trainRating(r, estimates, userFeatures, itemFeatures, implicitFeatures);
            }
        }

        // Wrap the user/item matrices because we won't use or modify them again
        return new SVDppModel(userFeatures,
                              itemFeatures,
                              implicitFeatures,
                              snapshot.userIndex(), snapshot.itemIndex(),
                              featureInfo);
    }


    protected void trainRating(IndexedPreference rating,
                               TrainingEstimator estimates,
                               RealMatrix userFeatures, RealMatrix itemFeatures, RealMatrix implicitFeatures) {

        final int uidx = rating.getUserIndex();
        final int iidx = rating.getItemIndex();

        // Use scratch vectors for each feature for better cache locality
        // Per-feature vectors are strided in the output matrices
        RealVector userFeatureVector = userFeatures.getRowVector(uidx); // user vector
        RealVector itemFeatureVector = itemFeatures.getRowVector(iidx); // item vector
        RealVector implicitFeatureVector = MatrixUtils.createRealVector(new double[featureCount]); // user-item profile vector

        // learning rates // TODO make customizable
        double learn_rate = 0.007;
        double reg_term = 0.015;

        // ratings for this user
        Collection<IndexedPreference> user_ratings = snapshot.getUserRatings(uidx);

        // calculate predicted rating
        double user_item_profile = getUserItemProfile(user_ratings, implicitFeatures,
                                                      userFeatureVector, itemFeatureVector, implicitFeatureVector);
        double estimate = estimates.get(rating);         // base score estimate
        double pred = estimate + user_item_profile;      // Calculate Prediction
        double rating_value = rating.getValue();         // Actual Rating Value
        double error = rating_value - pred;              // Calculate Error

        // compute delta in user vector : Pu <- Pu + learn_rate * (error * Qi - reg_term * Pu)
        RealVector uvec_deltas;
        uvec_deltas = itemFeatureVector.combine(error, -(reg_term), userFeatureVector);
        uvec_deltas.mapMultiplyToSelf(learn_rate);

        // compute delta in item vector : Qi <- Qi + learn_rate * (error * (Pu + |N(u)|^-1/2 * SUM Yj) - reg_term * Qi)
        RealVector ivec_deltas;
        ivec_deltas = implicitFeatureVector.combine(error, -(reg_term), itemFeatureVector);
        ivec_deltas.mapMultiplyToSelf(learn_rate);

        // compute deltas for implicit feedback vector : Yj <- Yj + learn_rate * ( error * |N(u)|^-1/2 * Qi - reg_term * Yj)
        RealVector temp_vec;
        int item_index;
        for(IndexedPreference ur : user_ratings){
            item_index = ur.getItemIndex();
            temp_vec = itemFeatureVector.combine(error * Math.pow(user_ratings.size(), -0.5), -(reg_term), implicitFeatures.getRowVector(item_index));
            temp_vec.mapMultiplyToSelf(learn_rate);
            temp_vec.combineToSelf(1, 1, implicitFeatures.getRowVector(item_index));
            implicitFeatures.setRowVector(item_index, temp_vec);
        }

        // apply deltas
        userFeatures.setRowVector(uidx, uvec_deltas.combineToSelf(1, 1, userFeatureVector));
        itemFeatures.setRowVector(iidx, ivec_deltas.combineToSelf(1, 1, itemFeatureVector));
    }


    /**
     * Calculates and returns User-Item Profile
     * user-item profile ->  Qi o (Pu + |N(u)|^-1/2 * SUM Yj)
     */
    protected double getUserItemProfile(Collection<IndexedPreference> user_ratings,
                                        RealMatrix implicitFeatures,
                                        RealVector userFeatureVector,
                                        RealVector itemFeatureVector,
                                        RealVector implicitFeatureVector){

        for(IndexedPreference ur : user_ratings){
            // TODO No function that adds to self, so I could only use combineToSelf, is there a better way? or should I just make a custom func?
            implicitFeatureVector.combineToSelf(1, 1, implicitFeatures.getRowVector(ur.getItemIndex()));
        }
        implicitFeatureVector.mapMultiplyToSelf(Math.pow(user_ratings.size(), -0.5));
        implicitFeatureVector.combineToSelf(1, 1, userFeatureVector);  // user item profile - (Pu + |N(u)|^-1/2 * SUM Yj)
        return itemFeatureVector.dotProduct(implicitFeatureVector);
    }

//
//    /**
//     * Train a feature using a collection of ratings.  This method iteratively calls {@link
//     * #doFeatureIteration(TrainingEstimator, Collection, RealVector, RealVector, RealVector, double)}  to train
//     * the feature.  It can be overridden to customize the feature training strategy.
//     *
//     * <p>We use the estimator to maintain the estimate up through a particular feature value,
//     * rather than recomputing the entire kernel value every time.  This hopefully speeds up training.
//     * It means that we always tell the updater we are training feature 0, but use a subvector that
//     * starts with the current feature.</p>
//     *
//     *
//     * @param feature   The number of the current feature.
//     * @param estimates The current estimator.  This method is <b>not</b> expected to update the
//     *                  estimator.
//     * @param userFeatureVector      The user feature values.  This has been initialized to the initial value,
//     *                  and may be reused between features.
//     * @param itemFeatureVector      The item feature values.  This has been initialized to the initial value,
//     *                  and may be reused between features.
//     * @param implicitFeatureVector The implicit feedback feature values. This has been initialized to the initial value,
//     *                  and may be reused between features.
//     * @param fib       The feature info builder. This method is only expected to add information
//     *                  about its training rounds to the builder; the caller takes care of feature
//     *                  number and summary data.
//     * @see #doFeatureIteration(TrainingEstimator, Collection, RealVector, RealVector, RealVector, double)
//     * @see #summarizeFeature(RealVector, RealVector, FeatureInfo.Builder)
//     */
//    protected void trainFeature(int feature, TrainingEstimator estimates,
//                                RealVector userFeatureVector, RealVector itemFeatureVector,
//                                RealVector implicitFeatureVector, FeatureInfo.Builder fib) {
//        double rmse = Double.MAX_VALUE;
//        double trail = initialValue * initialValue * (featureCount - feature - 1);
//        TrainingLoopController controller = rule.getTrainingLoopController();
//        Collection<IndexedPreference> ratings = snapshot.getRatings();
//        while (controller.keepTraining(rmse)) {
//            rmse = doFeatureIteration(estimates, ratings, userFeatureVector, itemFeatureVector, implicitFeatureVector, trail);
//            fib.addTrainingRound(rmse);
//            logger.trace("iteration {} finished with RMSE {}", controller.getIterationCount(), rmse);
//        }
//    }
//
//    /**
//     * Do a single feature iteration.
//     *
//     *
//     *
//     * @param estimates The estimates.
//     * @param ratings   The ratings to train on.
//     * @param userFeatureVector The user column vector for the current feature.
//     * @param itemFeatureVector The item column vector for the current feature.
//     * @param implicitFeatureVector The implicit feedback column vector for the current feature.
//     * @param trail The sum of the remaining user-item-feature values.
//     * @return The RMSE of the feature iteration.
//     */
//    protected double doFeatureIteration(TrainingEstimator estimates,
//                                        Collection<IndexedPreference> ratings,
//                                        RealVector userFeatureVector, RealVector itemFeatureVector,
//                                        RealVector implicitFeatureVector, double trail) {
//        // We'll create a fresh updater for each feature iteration
//        // Not much overhead, and prevents needing another parameter
//        SVDppUpdater updater = rule.createUpdater();
//
//        for (IndexedPreference r : ratings) {
//            final int uidx = r.getUserIndex();
//            final int iidx = r.getItemIndex();
//
//            updater.prepare(0, r.getValue(), estimates.get(r),
//                            userFeatureVector.getEntry(uidx), itemFeatureVector.getEntry(iidx),
//                            implicitFeatureVector, trail);
//
//            // Step 3: Update feature values
//            userFeatureVector.addToEntry(uidx, updater.getUserFeatureUpdate());
//            itemFeatureVector.addToEntry(iidx, updater.getItemFeatureUpdate());
//        }
//
//        return updater.getRMSE();
//    }
//
//    /**
//     * Add a feature's summary to the feature info builder.
//     *
//     * @param ufv The user values.
//     * @param ifv The item values.
//     * @param fib  The feature info builder.
//     */
//    protected void summarizeFeature(RealVector ufv, RealVector ifv, FeatureInfo.Builder fib) {
//        StatUtils.sum(ifv.toArray());
//
//        fib.setUserAverage(StatUtils.sum(ufv.toArray()) / ufv.getDimension())
//           .setItemAverage(StatUtils.sum(ifv.toArray()) / ifv.getDimension())
//           .setSingularValue(ufv.getNorm() * ifv.getNorm());
//    }
}
