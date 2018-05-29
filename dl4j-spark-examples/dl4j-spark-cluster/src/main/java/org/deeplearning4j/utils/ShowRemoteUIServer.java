package org.deeplearning4j.utils;


import org.deeplearning4j.ui.api.UIServer;

public class ShowRemoteUIServer {
    public static void main(String[] args) {
        UIServer uiServer = UIServer.getInstance();
        uiServer.enableRemoteListener();
    }
}
