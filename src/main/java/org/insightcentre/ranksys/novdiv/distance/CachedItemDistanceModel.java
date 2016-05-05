/*
 * Copyright (C) 2016 Insight Centre for Data Analytics http://insight-centre.org
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package org.insightcentre.ranksys.novdiv.distance;

import es.uam.eps.ir.ranksys.fast.index.FastItemIndex;
import es.uam.eps.ir.ranksys.novdiv.distance.ItemDistanceModel;

import java.util.Map;
import java.util.function.Function;
import java.util.function.ToDoubleFunction;
import java.util.stream.Collectors;

public class CachedItemDistanceModel<I> implements ItemDistanceModel<I> {

    private Map<I, Map<I, Double>> cachedDistances;

    public CachedItemDistanceModel(ItemDistanceModel<I> itemDistanceModel, FastItemIndex<I> iIndex) {
        this.cachedDistances = iIndex.getAllItems().parallel()
                .collect(Collectors.toMap(Function.identity(), i -> iIndex.getAllItems()
                        .collect(Collectors.toMap(Function.identity(), j -> {
                            double distance = itemDistanceModel.dist(i, j);
                            if (Double.isNaN(distance)) {
                                distance = 0.0;
                            }
                            return distance;
                        }))));
    }

    @Override
    public ToDoubleFunction<I> dist(I i) {
        return j -> dist(i, j);
    }

    @Override
    public double dist(I i, I j) {
        return cachedDistances.get(i).get(j);
    }
}