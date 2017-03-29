package org.deeplearning4j.emr;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.amazonaws.auth.AWSStaticCredentialsProvider;
import com.amazonaws.auth.BasicAWSCredentials;
import com.amazonaws.regions.Regions;
import com.amazonaws.services.elasticmapreduce.AmazonElasticMapReduce;
import com.amazonaws.services.elasticmapreduce.AmazonElasticMapReduceClientBuilder;
import com.amazonaws.services.elasticmapreduce.model.ActionOnFailure;
import com.amazonaws.services.elasticmapreduce.model.Application;
import com.amazonaws.services.elasticmapreduce.model.HadoopJarStepConfig;
import com.amazonaws.services.elasticmapreduce.model.JobFlowInstancesConfig;
import com.amazonaws.services.elasticmapreduce.model.RunJobFlowRequest;
import com.amazonaws.services.elasticmapreduce.model.RunJobFlowResult;
import com.amazonaws.services.elasticmapreduce.model.StepConfig;
import com.amazonaws.services.elasticmapreduce.util.StepFactory;
import com.amazonaws.services.s3.AmazonS3;
import com.amazonaws.services.s3.AmazonS3ClientBuilder;
import com.amazonaws.services.s3.model.PutObjectRequest;

public class EmrSparkExample {

    private static final Logger log          = LoggerFactory.getLogger(EmrSparkExample.class);

    private String              accessKey    = "yourawsaccesskey";
    private String              secretKey    = "yourawssecretkey";
    private Regions             region       = Regions.EU_CENTRAL_1;
    private boolean             debug        = false;
    private boolean             execute      = true;
    private boolean             upload       = true;
    private String              bucketName   = "com.your.bucket";
    private boolean             keepAlive    = true;
    private String              instanceType = "m3.xlarge";
    private File                uberJar      = new File("./target/dl4j-spark-0.8-SNAPSHOT-bin.jar");
    private String              className    = "org.deeplearning4j.stats.TrainingStatsExample";

    public static void main(String[] args) {
        EmrSparkExample test = new EmrSparkExample();
        test.start();
    }

    public void start() {
        AmazonElasticMapReduceClientBuilder builder = AmazonElasticMapReduceClientBuilder.standard();
        builder.withRegion(region);
        builder.withCredentials(getCredentialsProvider());

        AmazonElasticMapReduce emr = builder.build();

        List<StepConfig> steps = new ArrayList<>();

        if (upload) {
            log.info("uploading uber jar");

            AmazonS3ClientBuilder s3builder = AmazonS3ClientBuilder.standard();
            s3builder.withRegion(region);
            s3builder.withCredentials(getCredentialsProvider());
            AmazonS3 s3Client = s3builder.build();

            if (!s3Client.doesBucketExist(bucketName)) {
                s3Client.createBucket(bucketName);
            }

            s3Client.putObject(new PutObjectRequest(bucketName, uberJar.getName(), uberJar));
        }

        if (debug) {
            log.info("enable debug");

            StepFactory stepFactory = new StepFactory(builder.getRegion() + ".elasticmapreduce");
            StepConfig enableDebugging = new StepConfig().withName("Enable Debugging").withActionOnFailure(ActionOnFailure.TERMINATE_JOB_FLOW).withHadoopJarStep(stepFactory.newEnableDebuggingStep());
            steps.add(enableDebugging);
        }

        if (execute) {
            log.info("execute spark step");

            HadoopJarStepConfig sparkStepConf = new HadoopJarStepConfig();
            sparkStepConf.withJar("command-runner.jar");
            sparkStepConf.withArgs("spark-submit", "--deploy-mode", "cluster", "--class", className, getS3UberJarUrl(), "-useSparkLocal", "false");

            ActionOnFailure action = ActionOnFailure.TERMINATE_JOB_FLOW;

            if (keepAlive) {
                action = ActionOnFailure.CONTINUE;
            }

            StepConfig sparkStep = new StepConfig().withName("Spark Step").withActionOnFailure(action).withHadoopJarStep(sparkStepConf);
            steps.add(sparkStep);
        }

        log.info("create spark cluster");

        Application sparkApp = new Application().withName("Spark");

        RunJobFlowRequest request = new RunJobFlowRequest().withName("Spark Cluster").withSteps(steps).withServiceRole("EMR_DefaultRole").withJobFlowRole("EMR_EC2_DefaultRole")
                .withApplications(sparkApp).withReleaseLabel("emr-5.4.0").withLogUri(getS3BucketLogsUrl()).withInstances(new JobFlowInstancesConfig().withEc2KeyName("spark").withInstanceCount(5)
                        .withKeepJobFlowAliveWhenNoSteps(keepAlive).withMasterInstanceType(instanceType).withSlaveInstanceType(instanceType));

        RunJobFlowResult result = emr.runJobFlow(request);

        log.info(result.toString());

        log.info("done");
    }

    public String getS3UberJarUrl() {
        return getS3BucketUrl() + "/" + uberJar.getName();
    }

    public String getS3BucketUrl() {
        return "s3://" + bucketName;
    }

    public String getS3BucketLogsUrl() {
        return getS3BucketUrl() + "/logs";
    }

    public AWSStaticCredentialsProvider getCredentialsProvider() {
        return new AWSStaticCredentialsProvider(new BasicAWSCredentials(accessKey, secretKey));
    }

}
