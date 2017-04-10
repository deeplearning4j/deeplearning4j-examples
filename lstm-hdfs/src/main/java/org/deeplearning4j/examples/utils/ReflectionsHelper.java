package org.deeplearning4j.examples.utils;

import java.net.URL;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.reflections.vfs.Vfs;

import com.google.common.collect.Lists;

/**
 * Inspired heavily by
 * 
 * <link>https://git-wip-us.apache.org/repos/asf?p=isis.git;a=blob;f=core/applib/src/main/java/org/apache/isis/applib/
 * services/classdiscovery/ ClassDiscoveryServiceUsingReflections.java;h= 283f053ddb15bfe32f111d88891602820854415e;hb=
 * 283f053ddb15bfe32f111d88891602820854415e</link>
 */
public class ReflectionsHelper {

    /**
     * OSX contains file:// resources on the classpath including .mar and .jnilib files.
     * 
     * Reflections use of Vfs doesn't recognize these URLs and logs warns when it sees them. By registering those file
     * endings, we supress the warns.
     */

    public static void registerUrlTypes() {
        final List<Vfs.UrlType> urlTypes = Lists.newArrayList();

        // include a list of file extensions / filenames to be recognized
        urlTypes.add(new EmptyIfFileEndingsUrlType(".mar", ".jnilib", ".jar", "hive", ".pom"));
        // Vfs.addDefaultURLTypes(new BundleUrlType());
        urlTypes.addAll(Arrays.asList(Vfs.DefaultUrlTypes.values()));
        Vfs.setDefaultURLTypes(urlTypes);
    }

    private static class EmptyIfFileEndingsUrlType implements Vfs.UrlType {
        private final List<String> fileEndings;

        private EmptyIfFileEndingsUrlType(final String... fileEndings) {
            this.fileEndings = Lists.newArrayList(fileEndings);
        }

        public boolean matches(URL url) {
            final String protocol = url.getProtocol();
            final String externalForm = url.toExternalForm();
            if (!protocol.equals("file")) {
                return false;
            }
            for (String fileEnding : fileEndings) {
                if (externalForm.endsWith(fileEnding))
                    return true;
            }
            return false;
        }

        public Vfs.Dir createDir(final URL url) throws Exception {

            return emptyVfsDir(url);
        }

        private static Vfs.Dir emptyVfsDir(final URL url) {
            return new Vfs.Dir() {
                @Override
                public String getPath() {
                    return url.toExternalForm();
                }

                @Override
                public Iterable<Vfs.File> getFiles() {
                    return Collections.emptyList();
                }

                @Override
                public void close() {}
            };
        }
    }
}
