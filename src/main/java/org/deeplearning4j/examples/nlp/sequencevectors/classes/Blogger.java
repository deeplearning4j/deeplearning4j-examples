package org.deeplearning4j.examples.nlp.sequencevectors.classes;

import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

/**
 * @author raver119@gmail.com
 */
public class Blogger extends SequenceElement {
    @Getter @Setter private int id;

    public Blogger() {

    }

    public Blogger(int id) {
        this.id = id;
    }

    @Override
    public String getLabel() {
        return null;
    }

    @Override
    public String toJSON() {
        return null;
    }
}
