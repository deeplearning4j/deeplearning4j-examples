/*******************************************************************************
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

package org.deeplearning4j.examples.arbiter.genetic;

import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.SynchronizedRandomGenerator;
import org.deeplearning4j.arbiter.ComputationGraphSpace;
import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultReference;
import org.deeplearning4j.arbiter.optimize.generator.GeneticSearchCandidateGenerator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.Chromosome;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.CrossoverOperator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.KPointCrossover;
import org.deeplearning4j.arbiter.optimize.generator.genetic.crossover.parentselection.TwoParentSelection;
import org.deeplearning4j.arbiter.optimize.generator.genetic.culling.CullOperator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.culling.LeastFitCullOperator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationListener;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationModel;
import org.deeplearning4j.arbiter.optimize.generator.genetic.selection.GeneticSelectionOperator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.selection.SelectionOperator;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
import org.deeplearning4j.arbiter.scoring.impl.EvaluationScoreFunction;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.graph.ComputationGraph;

import java.util.List;

/**
 * In this hyperparameter optimization example, we change the default behavior of the genetic candidate generator.
 * All parts of the genetic candidate generator can be changed either by changing the parameters of the default
 * implementations, or by supplying your own implementation.
 * Here, we'll use an alternate parent selection and culling behaviors.
 *
 * @author Alexandre Boulanger
 */

public class CustomGeneticHyperparameterOptimizationExample {

    public static void main(String[] args) throws Exception {

        ComputationGraphSpace cgs = GeneticSearchExampleConfiguration.GetGraphConfiguration();

        EvaluationScoreFunction scoreFunction = new EvaluationScoreFunction(Evaluation.Metric.F1);

        // The ExampleCullOperator extends the default cull operator (least fit) to include an artificial predator.
        CullOperator cullOperator = new ExampleCullOperator();
        PopulationModel populationModel = new PopulationModel.Builder()
            .cullOperator(cullOperator)
            .build();

        // We'll use the k-point crossover with our custom implementation of the parent selection. Our implementation
        // will make sure that one of the parent is one of the best 5 candidate of the population.
        TwoParentSelection parentSelection = new ExampleParentSelection();
        CrossoverOperator crossoverOperator = new KPointCrossover.Builder()
            .parentSelection(parentSelection)
            .build() ;

        // The selection operator is what generates new candidates. The default implementation generates random candidates
        // when the population is below a certain size (the maximum size minus the cull ratio) and uses crossover + mutation
        // when enough candidates are in the population.
        // Here, we tell the default selection operator to use our crossover implementation.
        SelectionOperator selectionOperator = new GeneticSelectionOperator.Builder()
            .crossoverOperator(crossoverOperator)
            .build();

        // This is where we create the GeneticSearchCandidateGenerator with its default behavior:
        GeneticSearchCandidateGenerator candidateGenerator = new GeneticSearchCandidateGenerator.Builder(cgs, scoreFunction)
            .populationModel(populationModel)
            .selectionOperator(selectionOperator)
            .build();

        // Let's have a listener to print the population size after each evaluation.
        populationModel.addListener(new ExamplePopulationListener());

        IOptimizationRunner runner = GeneticSearchExampleConfiguration.BuildRunner(candidateGenerator, scoreFunction);

        //Start the hyperparameter optimization
        runner.execute();

        //Print out some basic stats regarding the optimization procedure
        String s = "Best score: " + runner.bestScore() + "\n" +
            "Index of model with best score: " + runner.bestScoreCandidateIndex() + "\n" +
            "Number of configurations evaluated: " + runner.numCandidatesCompleted() + "\n";
        System.out.println(s);


        //Get all results, and print out details of the best result:
        int indexOfBestResult = runner.bestScoreCandidateIndex();
        List<ResultReference> allResults = runner.getResults();

        OptimizationResult bestResult = allResults.get(indexOfBestResult).getResult();
        ComputationGraph bestModel = (ComputationGraph) bestResult.getResultReference().getResultModel();

        System.out.println("\n\nConfiguration of best model:\n");
        System.out.println(bestModel.getConfiguration().toJson());

    }

    // This is an example of a custom behavior for the genetic algorithm. We force one of the parent to be one of the
    // best 5 candidates of the population.
    public static class ExampleParentSelection extends TwoParentSelection {
        private final RandomGenerator rng = new SynchronizedRandomGenerator(new JDKRandomGenerator());

        @Override
        public double[][] selectParents() {
            double[][] parents = new double[2][];

            // Select the first parent among the best 5 of the population. The population is always sorted -- the lower
            // the index, the better is the candidate.
            int parent1Idx = rng.nextInt(5);

            // The other parent is anyone in the population except the one selected above
            int parent2Idx;
            do {
                parent2Idx = rng.nextInt(population.size());
            } while (parent1Idx == parent2Idx);

            parents[0] = population.get(parent1Idx).getGenes();
            parents[1] = population.get(parent2Idx).getGenes();

            return parents;
        }
    }

    // Here we extend the behavior of the default cull operator (least fit) to include an artificial predator.
    // When the population is culled, an additional random number (0, 1 or 2) of elements are removed. These elements
    // will be replaced with random candidates by the genetic selection operation.
    // This is probably not the best idea in terms of convergence but it is a good way to demonstrate how one can change
    // the genetic search algorithm's behavior
    public static class ExampleCullOperator extends LeastFitCullOperator {
        private final RandomGenerator rng = new SynchronizedRandomGenerator(new JDKRandomGenerator());

        @Override
        public void cullPopulation() {
            super.cullPopulation();

            // Remove between 0 and 2 candidates that falls prey to a 'predator'
            int preyCount = rng.nextInt(3);
            for(int i = 0; i < preyCount; ++i) {
                int preyIdx = rng.nextInt(population.size());
                population.remove(preyIdx);
            }
            System.out.println(String.format("Randomly removed %1$s candidate(s).", preyCount));
        }
    }

    public static class ExamplePopulationListener implements PopulationListener {

        @Override
        public void onChanged(List<Chromosome> population) {
            double best = population.get(0).getFitness();
            double average = population.stream()
                .mapToDouble(Chromosome::getFitness)
                .average()
                .getAsDouble();
            System.out.println(String.format("\nPopulation size is %1$s, best score is %2$s, average score is %3$s", population.size(), best, average));
        }
    }
}
