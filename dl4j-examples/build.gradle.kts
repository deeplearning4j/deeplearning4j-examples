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

import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    `java-library`
    id("application")
    id("kotlin")
}

allprojects {
    val kotlinVersion: String by extra("1.3.72")
    val dl4jVersion: String by extra("1.0.0-beta6")
    val slf4jVersion: String by extra("1.7.30")
}

repositories {
    maven("https://oss.sonatype.org/content/repositories/snapshots")
    jcenter()
}

group = "org.deeplearning4j.examples"
application {
    mainClassName = "org.deeplearning4j.examples.feedforward.mnist.MLPMnistSingleLayerExample"
}

tasks.withType<KotlinCompile>().all {
    kotlinOptions {
        jvmTarget = JavaVersion.VERSION_1_8.toString()
    }
}

sourceSets {
    main {
        java{
            srcDir("src/main/java")
            /**
             * Kotlin examples do not use java sources, so let's exclude them all.
             */
            exclude("**")
        }
    }
}

internal val singleLayerExample = tasks.register<JavaExec>("MLPMnistSingleLayerExample") {
    classpath = sourceSets.main.get().runtimeClasspath
    main = "org.deeplearning4j.examples.feedforward.mnist.MLPMnistSingleLayerExample"
}

internal val twoLayerExample = tasks.register<JavaExec>("MLPMnistTwoLayerExample") {
    classpath = sourceSets.main.get().runtimeClasspath
    main = "org.deeplearning4j.examples.feedforward.mnist.MLPMnistTwoLayerExample"
}

dependencies {
    val kotlinVersion = project.extra["kotlinVersion"].toString()
    val dl4jVersion = project.extra["dl4jVersion"].toString()
    val slf4jVersion = project.extra["slf4jVersion"].toString()

    implementation ("org.jetbrains.kotlin:kotlin-stdlib:$kotlinVersion")
    implementation ("org.deeplearning4j:deeplearning4j-core:$dl4jVersion")
    implementation ("org.nd4j:nd4j-native-platform:$dl4jVersion")
    implementation ("org.slf4j:slf4j-simple:$slf4jVersion")
    implementation ("org.slf4j:slf4j-api:$slf4jVersion")
}

/**
 * TODO More Java examples to be converted to Kotlin.
 */
