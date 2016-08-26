package org.deeplearning4j.examples.misc.customlayers;

import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;

/**
 * Created by Alex on 26/08/2016.
 */
public class CustomLayer extends FeedForwardLayer {

    private String secondActivationFunction;

    private CustomLayer(Builder builder){
        super(builder);
        this.secondActivationFunction = secondActivationFunction;
    }

    public String getSecondActivationFunction(){
        return secondActivationFunction;
    }




    //Here's an implementation of a builder pattern, to allow us to easily configure the layer
    public static class Builder extends FeedForwardLayer.Builder<Builder>{

        private String secondActivationFunction;

        //This is an example of a custom property in the configuration
        public Builder secondActivationFunction(String secondActivationFunction){
            this.secondActivationFunction = secondActivationFunction;
            return this;
        }

        @Override
        @SuppressWarnings("unchecked")  //To stop warnings about unchecked cast. Not required.
        public CustomLayer build() {
            return new CustomLayer(this);
        }
    }

}
