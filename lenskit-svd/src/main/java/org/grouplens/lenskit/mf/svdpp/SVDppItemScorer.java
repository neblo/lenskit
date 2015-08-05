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
import mikera.matrixx.Matrix;
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
//    protected final BiasedMFKernel kernel;
    private UserEventDAO dao;
    private final ItemScorer baselineScorer;
    private final int featureCount;

    @Nullable
//    private final SVDppUpdateRule rule;

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
//        this.rule = rule;

//        if (dom == null) {
//            kernel = new DotProductKernel();
//        } else {
//            kernel = new DomainClampingKernel(dom);
//        }

        featureCount = model.getFeatureCount();
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
        assert userFV != null;

        MutableSparseVector estimates = initialEstimates(user, ratings, scores.keySet());
        // propagate estimates to the output scores
        scores.set(estimates);

        if (!ratings.isEmpty()) {

            for (VectorEntry score : scores ){
                final long item = score.getKey();

                RealVector itemFV = model.getItemVector(item); // TODO score.getKey() gets item's key right (not position in vector)?
                assert itemFV != null;
                RealVector implicitFV = MatrixUtils.createRealVector(new double[featureCount]);

                // Calculate user-item profile ->  Qi o (Pu + |N(u)|^-1/2 * SUM Yj)
                RealVector temp_vec;
                for(VectorEntry ur : scores){
                    temp_vec = model.getImplicitFeedbackVector(ur.getKey());
                    implicitFV.combineToSelf(1, 1, temp_vec);
                }
                implicitFV.mapMultiplyToSelf(Math.pow(ratings.size(), -0.5));
                implicitFV.add(userFV);  // user item profile - (Pu + |N(u)|^-1/2 * SUM Yj)

                double user_item_profile = itemFV.dotProduct(implicitFV);
                double estimate = estimates.get(item);  // base score estimate
                double pred = estimate + user_item_profile;   // Calculate Prediction

                scores.set(score, pred);
            }
        }
    }
}
