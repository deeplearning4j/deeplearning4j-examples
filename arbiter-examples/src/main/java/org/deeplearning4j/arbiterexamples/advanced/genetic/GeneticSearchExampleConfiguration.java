/* *****************************************************************************
 * Copyright (c) 2020 Konduit K.K.
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.arbiterexamples.advanced.genetic;

import org.deeplearning4j.arbiter.ComputationGraphSpace;
import org.deeplearning4j.arbiter.conf.updater.AdamSpace;
import org.deeplearning4j.arbiter.conf.updater.schedule.StepScheduleSpace;
import org.deeplearning4j.arbiter.layers.DenseLayerSpace;
import org.deeplearning4j.arbiter.layers.OutputLayerSpace;
import org.deeplearning4j.arbiter.optimize.api.CandidateGenerator;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.api.data.DataSource;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultSaver;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxCandidatesCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.TerminationCondition;
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.discrete.DiscreteParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.integer.IntegerParameterSpace;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.LocalOptimizationRunner;
import org.deeplearning4j.arbiter.saver.local.FileModelSaver;
import org.deeplearning4j.arbiter.task.ComputationGraphTaskCreator;
import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.ScheduleType;

import java.io.File;
import java.util.Properties;

class GeneticSearchExampleConfiguration {

    static ComputationGraphSpace GetGraphConfiguration() {
        int inputSize = 784;
        int outputSize = 47;

        // First, we setup the hyperspace parameters. These are the values which will change, breed and mutate
        // while attempting to find the best candidate.
        DiscreteParameterSpace<Activation> activationSpace = new DiscreteParameterSpace<>(Activation.ELU,
            Activation.RELU,
            Activation.LEAKYRELU,
            Activation.TANH,
            Activation.SELU,
            Activation.HARDSIGMOID);
        IntegerParameterSpace[] layersParametersSpace = new IntegerParameterSpace[] {
            new IntegerParameterSpace(outputSize, inputSize),
            new IntegerParameterSpace(outputSize, inputSize),
        };
        ParameterSpace<IUpdater> updaterSpace = AdamSpace.withLRSchedule(new StepScheduleSpace(ScheduleType.EPOCH, new ContinuousParameterSpace(0.0, 0.1), 0.5, 2));
        ParameterSpace<Double> l2Space = new ContinuousParameterSpace(0.0, 0.01);
        ParameterSpace<Double> dropoutSpace = new ContinuousParameterSpace(0.0, 1.0);

        // Then we plug our hyperspace parameters in our model configuration -- a dense layer net with an input layer,
        // 2 hidden layers and an output layer.
        ComputationGraphSpace.Builder builder = new ComputationGraphSpace.Builder()
            .seed(123)
            .activation(activationSpace)
            .weightInit(WeightInit.XAVIER)
            .updater(updaterSpace)
            .l2(l2Space)
            .dropOut(dropoutSpace)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .addInputs("in")
            .addLayer("0",new DenseLayerSpace.Builder().nIn(inputSize).nOut(layersParametersSpace[0]).build(),"in");
        for(int i = 1; i < layersParametersSpace.length; ++i) {
            builder = builder.addLayer(String.valueOf(i), new DenseLayerSpace.Builder().nIn(layersParametersSpace[i-1]).nOut(layersParametersSpace[i]).build(), String.valueOf(i - 1));
        }
        builder = builder.addLayer("out", new OutputLayerSpace.Builder()
            .nIn(layersParametersSpace[layersParametersSpace.length - 1])
            .nOut(outputSize)
            .lossFunction(LossFunctions.LossFunction.MCXENT)
            .activation(Activation.SOFTMAX)
            .build(), String.valueOf(layersParametersSpace.length - 1))
            .setOutputs("out")
            .backpropType(BackpropType.Standard)
            .numEpochs(10);
        return builder.build();
    }

    public static IOptimizationRunner BuildRunner(CandidateGenerator candidateGenerator, ScoreFunction scoreFunction) {
        Class<? extends DataSource> dataSourceClass = ExampleDataSource.class;
        Properties dataSourceProperties = new Properties();
        dataSourceProperties.setProperty("minibatchSize", "1024");

        TerminationCondition[] terminationConditions = {
            new MaxCandidatesCondition(100)};

        String baseSaveDirectory = "arbiterExample/";
        File f = new File(baseSaveDirectory);
        if (f.exists()) f.delete();
        f.mkdir();
        ResultSaver modelSaver = new FileModelSaver(baseSaveDirectory);

        OptimizationConfiguration configuration = new OptimizationConfiguration.Builder()
            .candidateGenerator(candidateGenerator)
            .dataSource(dataSourceClass,dataSourceProperties)
            .scoreFunction(scoreFunction)
            .terminationConditions(terminationConditions)
            .modelSaver(modelSaver)
            .build();

        return new LocalOptimizationRunner(configuration, new ComputationGraphTaskCreator());
    }

    public static class ExampleDataSource implements DataSource {
        private int minibatchSize;

        public ExampleDataSource() {

        }

        @Override
        public void configure(Properties properties) {
            this.minibatchSize = Integer.parseInt(properties.getProperty("minibatchSize", "16"));
        }

        @Override
        public Object trainData() {
            try {
                return new EmnistDataSetIterator(EmnistDataSetIterator.Set.BALANCED, minibatchSize, true, 12345);

            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        @Override
        public Object testData() {
            try {
                return new EmnistDataSetIterator(EmnistDataSetIterator.Set.BALANCED, minibatchSize, false, 12345);

            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        @Override
        public Class<?> getDataType() {
            return DataSetIterator.class;
        }
    }
}
