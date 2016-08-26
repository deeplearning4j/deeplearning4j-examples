package org.deeplearning4j.examples.misc.customlayers;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.BaseLayer;

/**
 * Created by Alex on 26/08/2016.
 */
public class CustomLayerImpl extends BaseLayer<CustomLayer> {
    public CustomLayerImpl(NeuralNetConfiguration conf) {
        super(conf);
    }



}
