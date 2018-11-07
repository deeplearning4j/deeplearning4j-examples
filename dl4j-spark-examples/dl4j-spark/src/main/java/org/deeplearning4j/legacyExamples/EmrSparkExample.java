package org.deeplearning4j.legacyExamples;

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
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;

public class EmrSparkExample {

    private static final Logger log = LoggerFactory.getLogger(EmrSparkExample.class);

    // access- and secretkey by creating an IAM user in aws console, user needs
    // to granted AmazonElasticMapReduceFullAccess and AmazonEC2FullAccess
    // policy

    @Parameter(names = "-accessKey", description = "your aws accesskey")
    private String accessKey = "your_aws_user_accesskey";

    @Parameter(names = "-secretKey", description = "your aws secretKey")
    private String secretKey = "your_aws_user_secretKey";

    @Parameter(names = "-region", description = "your aws region")
    private Regions region = Regions.EU_CENTRAL_1;

    @Parameter(names = "-debug", description = "enable spark debug mode")
    private boolean debug = false;

    @Parameter(names = "-execute", description = "run spark step after cluster creation")
    private boolean execute = true;

    @Parameter(names = "-upload", description = "upload uber jar")
    private boolean upload = true;

    // default emr version runs spark2, so make sure to update spark version in
    // parent pom (default is spark1)

    @Parameter(names = "-emrVersion", description = "version of aws emr software stack")
    private String emrVersion = "emr-5.4.0";

    @Parameter(names = "-bucketName", description = "aws s3 bucket name for logs and uber jar")
    private String bucketName = "com.your.bucket";

    @Parameter(names = "-keepAlive", description = "keep cluster alive after execution of spark step")
    private boolean keepAlive = true;

    @Parameter(names = "-instanceType", description = "emr master and slave instance type")
    private String instanceType = "m3.xlarge";

    @Parameter(names = "-instanceCount", description = "instance count, 1 will be master and rest slaves")
    private int instanceCount = 3;

    @Parameter(names = "-uberJar", description = "uber jar with spark step to execute")
    private String uberJar = "./target/dl4j-spark-0.8-SNAPSHOT-bin.jar";

    @Parameter(names = "-className", description = "class name of spark step to execute")
    private String className = "org.deeplearning4j.stats.TrainingStatsExample";

    public static void main(String[] args) {
        EmrSparkExample test = new EmrSparkExample();
        test.entryPoint(args);
    }

    public void entryPoint(String[] args) {
        JCommander jcmdr = new JCommander(this);
        try {
            jcmdr.parse(args);
        } catch (ParameterException e) {
            jcmdr.usage();
            try {
                Thread.sleep(500);
            } catch (Exception e2) {
            }
            throw e;
        }

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

            File uberJarFile = new File(uberJar);

            s3Client.putObject(new PutObjectRequest(bucketName, uberJarFile.getName(), uberJarFile));
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

        // service and job flow role will be created automatically when
        // launching cluster in aws console, better do that first or create
        // manually

        RunJobFlowRequest request = new RunJobFlowRequest().withName("Spark Cluster").withSteps(steps).withServiceRole("EMR_DefaultRole").withJobFlowRole("EMR_EC2_DefaultRole")
                .withApplications(sparkApp).withReleaseLabel(emrVersion).withLogUri(getS3BucketLogsUrl()).withInstances(new JobFlowInstancesConfig().withEc2KeyName("spark")
                        .withInstanceCount(instanceCount).withKeepJobFlowAliveWhenNoSteps(keepAlive).withMasterInstanceType(instanceType).withSlaveInstanceType(instanceType));

        RunJobFlowResult result = emr.runJobFlow(request);

        log.info(result.toString());

        log.info("done");
    }

    public String getS3UberJarUrl() {
        return getS3BucketUrl() + "/" + new File(uberJar).getName();
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
