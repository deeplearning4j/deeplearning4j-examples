package org.deeplearning4j.examples.arbiter.genetic;

import org.deeplearning4j.arbiter.ComputationGraphSpace;
import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultReference;
import org.deeplearning4j.arbiter.optimize.generator.GeneticSearchCandidateGenerator;
import org.deeplearning4j.arbiter.optimize.generator.genetic.Chromosome;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationListener;
import org.deeplearning4j.arbiter.optimize.generator.genetic.population.PopulationModel;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
import org.deeplearning4j.arbiter.scoring.impl.EvaluationScoreFunction;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.graph.ComputationGraph;

import java.util.List;

/**
 * This is a basic hyperparameter optimization example using the genetic candidate generator of Arbiter to conduct a search.
 * This example illustrate how use the GeneticSearchCandidateGenerator with its default behavior: In short, it will
 * initially generate 20 random candidates and then start breeding (crossover genes of two parents and mutate some genes) candidates together. Then, when the population
 * hits its default max size (30), the population is culled back to 20 before adding the new candidate.
 *
 * @author Alexandre Boulanger
 */

public class BaseGeneticHyperparameterOptimizationExample {

    public static void main(String[] args) throws Exception {

        ComputationGraphSpace cgs = GeneticSearchExampleConfiguration.GetGraphConfiguration();

        EvaluationScoreFunction scoreFunction = new EvaluationScoreFunction(Evaluation.Metric.F1);

        // This is where we create the GeneticSearchCandidateGenerator with its default behavior:
        //  - a population that fits 30 candidates and is culled back to 20 when it overflows
        //  - new candidates are generated with a probability of 85% of being the result of breeding (a k-point crossover with 1 to 4 points)
        //  - the new candidate have a probability of 0.5% of sustaining a random mutation on one of its genes.
        GeneticSearchCandidateGenerator candidateGenerator = new GeneticSearchCandidateGenerator.Builder(cgs, scoreFunction).build();

        // Let's have a listener to print the population size after each evaluation.
        PopulationModel populationModel = candidateGenerator.getPopulationModel();
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

    public static class ExamplePopulationListener implements PopulationListener {

        @Override
        public void onChanged(List<Chromosome> population) {
            double best = population.get(0).getFitness();
            double average = population.stream()
                .mapToDouble(c -> c.getFitness())
                .average()
                .getAsDouble();
            System.out.println(String.format("\nPopulation size is %1$s, best score is %2$s, average score is %3$s", population.size(), best, average));
        }
    }
}
