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
import java.io.IOException;
import java.util.*;

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
        double range_min = 0.0001;
        double range_max = 0.1;
        double range_offset = range_min + (range_max - range_min);

        int userCount = snapshot.getUserIds().size();
        RealMatrix userFeatures = MatrixUtils.createRealMatrix(userCount, featureCount);
        for (int i = 0; i < userCount; i++) {
            for (int j = 0; j < featureCount; j++) {
                userFeatures.setEntry(i, j, rand.nextDouble() * range_offset);
            }
        }

        // setup itemFeatures ( set random value from 0.00001 to ~0.1 )
        int itemCount = snapshot.getItemIds().size();
        RealMatrix itemFeatures = MatrixUtils.createRealMatrix(itemCount, featureCount);
        for (int i = 0; i < itemCount; i++) {
            for (int j = 0; j < featureCount; j++) {
                itemFeatures.setEntry(i, j, rand.nextDouble() * range_offset);
            }
        }

        // setup implicitFeatures ( set random value from 0.00001 to ~0.1 )
        RealMatrix implicitFeatures = MatrixUtils.createRealMatrix(itemCount, featureCount);
        for (int i = 0; i < itemCount; i++) {
            for (int j = 0; j < featureCount; j++) {
                implicitFeatures.setEntry(i, j, rand.nextDouble() * range_offset);
            }
        }

        logger.debug("Learning rate is {}", rule.getLearningRate());
        logger.debug("Regularization term is {}", rule.getTrainingRegularization());

        logger.info("Building SVD with {} features for {} ratings",
                    featureCount, snapshot.getRatings().size());

        TrainingEstimator estimates = rule.makeEstimator(snapshot);

        // get ratings
        Collection<IndexedPreference> ratings = snapshot.getRatings();

        int MAX_ITERATIONS = 40;

        for (int i = 0; i < MAX_ITERATIONS; i++){
            StopWatch timer = new StopWatch();
            timer.start();
            // train model (one epoch)
            for (IndexedPreference r : ratings) {
                trainRating(r, estimates, userFeatures, itemFeatures, implicitFeatures);
                /////////// DEBUG //////////////
//                System.out.println("   ########### i = " + i + " rating = " + r.getValue() + "  / " + userFeatures.getEntry(0,0));
//                System.out.println("U " + userFeatures.getRowVector(0));
                /////////////////////////////////
            }
//            System.out.println("I[3] " + itemFeatures.getRowVector(3)); /// DEBUG ////
            timer.stop();
            System.out.println("Epoc  " + i + " in " + timer + " seconds");
        }
        /////// DEBUG ///////////
//        System.out.println("User " + userFeatures.getRowVector(0).toString());
//        System.out.println("Item " + itemFeatures.getRowVector(0).toString());
//        System.out.println("Impl " + implicitFeatures.getRowVector(0).toString());
        /////////////////////////
        // Wrap the user/item matrices because we won't use or modify them again
        return new SVDppModel(userFeatures,
                              itemFeatures,
                              implicitFeatures,
                              snapshot.userIndex(),
                              snapshot.itemIndex());
    }

    /**
     *
     * @param rating
     * @param estimates
     * @param userFeatures
     * @param itemFeatures
     * @param implicitFeatures
     */
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
        Collection<IndexedPreference> user_ratings = snapshot.getUserRatings(rating.getUserId());

        // Calculate user-item profile ->  Qi o (Pu + |N(u)|^-1/2 * SUM Yj)
        for (IndexedPreference ur : user_ratings) {
            // TODO No function that adds to self, so I could only use combineToSelf, is there a better way? or should I just make a custom func?
            implicitFeatureVector.combineToSelf(1, 1, implicitFeatures.getRowVector(ur.getItemIndex()));
        }
        implicitFeatureVector.mapMultiplyToSelf(Math.pow(user_ratings.size(), -0.5));
        implicitFeatureVector.combineToSelf(1, 1, userFeatureVector);  // user item profile - (Pu + |N(u)|^-1/2 * SUM Yj)
        double user_item_profile = itemFeatureVector.dotProduct(implicitFeatureVector);

        double estimate = estimates.get(rating);         // base score estimate
        double pred = estimate + user_item_profile;      // Calculate Prediction
        double rating_value = rating.getValue();         // Actual Rating Value
        double error = rating_value - pred;              // Calculate Error

        // compute delta in user vector : Pu <- Pu + learn_rate * (error * Qi - reg_term * Pu)
        RealVector uvec_deltas;
        uvec_deltas = itemFeatureVector.combine(error, -(reg_term), userFeatureVector);
        RealVector debug2 = uvec_deltas;//DEBUG /////////////////////////////////////////////
        uvec_deltas.mapMultiplyToSelf(learn_rate);
        RealVector debug3 = uvec_deltas;//DEBUG /////////////////////////////////////////////


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
        //// DEBUG ////
//        if (!(userFeatures.getRowVector(uidx).getEntry(1) < 1 && userFeatures.getRowVector(uidx).getEntry(1) > -1)){ ////// DEBUG
//            System.out.println("----- DEBUG -----\nUser_Item_profile : " + user_item_profile + "\nEstimate : " + estimate  + "\nPrediction : " + pred + "\nrating_value : " + rating_value + "\nerror : " + error);
//            System.out.println("2 : " + debug2);
//            System.out.println("3 : " + debug3);
//            System.out.println("User Ratings : " + user_ratings.toString());
//            System.out.println("Final " + userFeatures.getRowVector(uidx));
//            System.out.println("Item FV " + itemFeatureVector);
//            System.out.println("User FV " + userFeatureVector);
//            try {
//                int x = System.in.read();
//            } catch (IOException e) {
//                e.printStackTrace();
//            }
//        }
        ///////////////
        itemFeatures.setRowVector(iidx, ivec_deltas.combineToSelf(1, 1, itemFeatureVector));
        ////// DEBUG /////
//        if (!(itemFeatures.getRowVector(iidx).getEntry(1) < 1 && itemFeatures.getRowVector(iidx).getEntry(0) > -1 )){ ////// DEBUG
//            System.out.println(itemFeatures.getRowVector(iidx));
//            try {
//                int x = System.in.read();
//            } catch (IOException e) {
//                e.printStackTrace();
//            }
//
//        }
        //////////////////
    }
}
