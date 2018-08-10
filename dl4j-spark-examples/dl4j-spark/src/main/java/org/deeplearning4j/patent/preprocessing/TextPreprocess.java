package org.deeplearning4j.patent.preprocessing;

import java.text.Normalizer;
import java.util.List;

/**
 * Performs some very basic textual transformations
 * such as word shape, lower casing, and stripping of punctuation
 *
 * Pulled from InputHomogenization
 * @author Adam Gibson
 *
 */
public class TextPreprocess {
    private String input;
    private List<String> ignoreCharactersContaining;
    private boolean preserveCase;

    /**
     * Input text to applyTransformToOrigin
     * @param input the input text to applyTransformToOrigin,
     * equivalent to calling this(input,false)
     * wrt preserving case
     */
    public TextPreprocess(String input) {
        this(input,false);
    }

    /**
     *
     * @param input the input to applyTransformToOrigin
     * @param preserveCase whether to preserve case
     */
    public TextPreprocess(String input, boolean preserveCase) {
        this.input = input;
        this.preserveCase = preserveCase;
    }

    /**
     *
     * @param input the input to applyTransformToOrigin
     * @param ignoreCharactersContaining ignore transformation of words
     * containigng specified strings
     */
    public TextPreprocess(String input, List<String> ignoreCharactersContaining) {
        this.input = input;
        this.ignoreCharactersContaining = ignoreCharactersContaining;
    }
    /**
     * Returns the normalized text passed in via constructor
     * @return the normalized text passed in via constructor
     */
    public String transform() {
        StringBuilder sb = new StringBuilder();
        for(int i = 0; i < input.length(); i++) {
            if(ignoreCharactersContaining != null && ignoreCharactersContaining.contains(String.valueOf(input.charAt(i))))
                sb.append(input.charAt(i));
            else if(Character.isUpperCase(input.charAt(i)) && !preserveCase)
                sb.append(Character.toLowerCase(input.charAt(i)));
            else
                sb.append(input.charAt(i));

        }

        String normalized = Normalizer.normalize(sb.toString(), Normalizer.Form.NFD);
        normalized = normalized.replace("."," ");
        normalized = normalized.replace(","," ");
        normalized = normalized.replaceAll("\""," ");
        normalized = normalized.replace("'"," ");
        normalized = normalized.replace("("," ");
        normalized = normalized.replace(")"," ");
        normalized = normalized.replace("“"," ");
        normalized = normalized.replace("”"," ");
        normalized = normalized.replace("…"," ");
        normalized = normalized.replace("|"," ");
        normalized = normalized.replace("/"," ");
        normalized = normalized.replace("\\", " ");
        normalized = normalized.replace("[", " ");
        normalized = normalized.replace("]", " ");
        normalized = normalized.replace("‘"," ");
        normalized = normalized.replace("’"," ");
        //normalized = normalized.replace("-"," ");
        normalized = normalized.replaceAll("[!]+","!");
        return normalized;
    }

}
