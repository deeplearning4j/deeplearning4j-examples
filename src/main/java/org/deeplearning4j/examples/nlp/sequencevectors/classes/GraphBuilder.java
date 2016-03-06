package org.deeplearning4j.examples.nlp.sequencevectors.classes;

import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.util.ClassPathResource;
import org.canova.api.writable.Writable;
import org.deeplearning4j.models.sequencevectors.graph.primitives.Graph;
import org.deeplearning4j.models.sequencevectors.graph.primitives.IGraph;

import java.io.File;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * @author raver119@gmail.com
 */
public class GraphBuilder {

    /**
     * This method builds graph out of csv files
     * @return
     */
    public static Graph<Blogger, Void> buildGraph() throws Exception {
        // load vertexes
        File nodes = new ClassPathResource("/BlogCatalog/nodes.csv").getFile();

        CSVRecordReader reader = new CSVRecordReader(0,",");
        reader.initialize(new FileSplit(nodes));

        List<Blogger> bloggers = new ArrayList<>();
        while (reader.hasNext()) {
            List<Writable> lines = new ArrayList<>(reader.next());
            Blogger blogger = new Blogger(lines.get(0).toInt());
            bloggers.add(blogger);
        }

        reader.close();

        Graph<Blogger, Void> graph = new Graph<Blogger, Void>(bloggers, true);

        // load edges
        File edges = new ClassPathResource("/BlogCatalog/edges.csv").getFile();

        reader = new CSVRecordReader(0,",");
        reader.initialize(new FileSplit(edges));

        while (reader.hasNext()) {
            List<Writable> lines = new ArrayList<>(reader.next());
            int from = lines.get(0).toInt();
            int to = lines.get(1).toInt();

            graph.addEdge(from-1, to-1, null, false);
        }

        return graph;
    }
}
