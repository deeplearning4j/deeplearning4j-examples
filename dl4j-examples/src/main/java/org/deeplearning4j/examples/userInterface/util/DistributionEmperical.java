package org.deeplearning4j.examples.userInterface.util;

/**
 * Created by Don Smith on 3/27/2017.
 */
import java.text.NumberFormat;

public class DistributionEmperical {
    public static NumberFormat numberFormat = NumberFormat.getInstance();
    static {
        numberFormat.setMaximumFractionDigits(3);
        numberFormat.setMinimumFractionDigits(3);
    }
    private double m_leftBound;
    private double m_rightBound;
    private int m_bucketCount;
    private int m_buckets[];
    private double m_delta;
    private double m_total = 0.0;
    private int m_count = 0;

     public DistributionEmperical(double [] values, int count) {
        double min=Double.MAX_VALUE;
        double max=Double.NEGATIVE_INFINITY;
        for(double d:values) {
            if (d>max) {max=d;}
            if (d<min) {min=d;}
        }
        m_leftBound=min;
        m_rightBound= max;
        m_bucketCount=count;
        m_buckets = new int[count];
        m_delta = (m_rightBound - m_leftBound) / count;
        for(double d:values) {
            add(d);
        }
    }
    public DistributionEmperical(double left, double right, int count) {
        if (right <= left) {
            throw new IllegalArgumentException("right=" + right + " < left = " + left);
        }
        if (count <= 0) {
            throw new IllegalArgumentException("count = " + count + " <= 0");
        }
        m_leftBound = left;
        m_rightBound = right;
        m_bucketCount = count;
        m_buckets = new int[count];
        m_delta = (right - left) / count;
    }

    public void add(double d) {
        int index = (int) Math.floor((d - m_leftBound) / m_delta);
        if (index < 0) {
            index = 0;
        }
        if (index >= m_bucketCount) {
            index = m_bucketCount - 1;
        }
        m_buckets[index]++;
        m_count++;
        m_total += d;
    }

    private double getPercentile(double percentile) {
        double ratio= 0.01*percentile;
        double sum=0.0;
        int index=0;
        double value=m_leftBound;
        while (index<m_buckets.length) {
            sum+= m_buckets[index];
            index++;
            value+= m_delta;
            if (sum/m_count>= ratio) {
                break;
            }
        }
        return value;
    }
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        String minString = numberFormat.format(m_leftBound);
        String maxString = numberFormat.format(m_rightBound);
        double avg = m_total / m_count;
        String avgString = numberFormat.format(avg);
        sb.append("Distribution from " + m_leftBound + " to " + m_rightBound + " (" + m_bucketCount
            + " buckets):\n  min=" + minString + ", avg = " + avgString
            + ", 75P = " + numberFormat.format(getPercentile(75))
            + ", 95P = " + numberFormat.format(getPercentile(95))
            + ", 99P = " + numberFormat.format(getPercentile(99))
            + ", max= " + maxString + "\n");
        double value = m_leftBound;
        int cumulativeCount = 0;
        for (int i = 0; i < m_bucketCount; i++) {
            double newValue = value + m_delta;
            String to = " to " + numberFormat.format(newValue) + ":  " + m_buckets[i];
            if (i == m_bucketCount - 1) {
                to = " to       :  " + m_buckets[i];
            }
            sb.append(" " + numberFormat.format(value) + to);

            cumulativeCount += m_buckets[i];
            double percent = (100.0 * m_buckets[i]) / m_count;
            double percentCumulative = (100.0 * cumulativeCount) / m_count;
            sb.append("  (" + numberFormat.format(percent) + "%, "
                + numberFormat.format(percentCumulative) + "%)\n");
            value = newValue;
        }
        return sb.toString();
    }
    public int getCount() {
        return m_count;
    }
}
