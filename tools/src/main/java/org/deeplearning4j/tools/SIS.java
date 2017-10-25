package org.deeplearning4j.tools;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintStream;
import java.io.Writer;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;




/**
 * show informations in console
 * if required save informations in file
 * 
 *
 * @author clavvis 
 */

public class SIS {
	// System Informations Saving
	//
	private String  baseModulCode = "SIS";
	private String  modulCode     = "?";
	//
	private PrintStream out;
	@SuppressWarnings("unused")
	private PrintStream err;
	//
	private boolean wasOpenedFile = false;
	private boolean wasClosedFile = false;
	//
	private File    sis_File;
	private Writer  sis_Writer;
	//
	private int     writerErrorInfoCount = 0;
	private int     closedFileInfoCount  = 0;
	//
	//
	
	public void initValues(
			int mtLv,
			String superiorModulCode,
			PrintStream out,
			PrintStream err
			) {
		//
		mtLv ++;
		//
		modulCode = superiorModulCode + "." + baseModulCode;
		//
		this.out = out;
		this.err = err;
		//
	}
	
	public void initValues(
			int mtLv,
			String superiorModulCode,
			PrintStream out,
			PrintStream err,
			String fileDrcS,
			String base_FileCode,
			String part_FileCode,
			boolean ShowBriefInfo,
			boolean ShowFullInfo
			) {
		//
		mtLv ++;
		//
		modulCode = superiorModulCode + "." + baseModulCode;
		//
		String methodName = modulCode + "." + "initValues";
		//
		this.out = out;
		this.err = err;
		//
		if ( ShowBriefInfo || ShowFullInfo ) {
	    	out.format( "" );
	    	out.format( BTools.getMtLvESS( mtLv ) );
			out.format( methodName + ": " );
			out.format( "fileDrcS: " + fileDrcS + "; " );
			out.format( "base_FileCode: " + base_FileCode + "; " );
			out.format( "part_FileCode: " + part_FileCode + "; " );
//			out.format( "STm: %s; ", Tools.getSDatePM( System.currentTimeMillis(), "HH:mm:ss" ) + "; " );
			out.format( "%s", BTools.getSLcDtTm() );
			out.format( "%n" );
		}
		//
		initFile( mtLv, fileDrcS, base_FileCode, part_FileCode, ShowBriefInfo, ShowFullInfo );
		//
	}
	
	private void initFile(
			int mtLv,
			String fileDrcS,
			String base_FileCode,
			String part_FileCode,
			boolean ShowBriefInfo,
			boolean ShowFullInfo
			) {
		//
		mtLv ++;
		//
		String OInfo = "";
		//
		String methodName = modulCode + "." + "initFile";
		//
		if ( ShowBriefInfo || ShowFullInfo ) {
	    	out.format( "" );
	    	out.format( BTools.getMtLvESS( mtLv ) );
			out.format( methodName + ": " );
			out.format( "fileDrcS: " + fileDrcS + "; " );
			out.format( "base_FileCode: " + base_FileCode + "; " );
			out.format( "part_FileCode: " + part_FileCode + "; " );
			out.format( "%s", BTools.getSLcDtTm() );
			out.format( "%n" );
		}
		//
		part_FileCode = part_FileCode.replace( ":", "" );
		part_FileCode = part_FileCode.replace( "/", "" );
		part_FileCode = part_FileCode.replace( ".", "" );
		//
		File fileDrc  = new File( fileDrcS );
		//
		if ( !fileDrc.exists() ) {
			fileDrc.mkdirs();
			//
			out.format( "" );
			out.format( BTools.getMtLvESS( mtLv ) );
			out.format( methodName + ": " );
			out.format( "fileDrcS: %s; ", fileDrcS );
			out.format( "Directory was created; " );
			out.format( "%s", BTools.getSLcDtTm() );
			out.format( "%n" );
		}
		//
    	LocalDateTime LDT = LocalDateTime.now();
		//
    	String TimeS = LDT.format( DateTimeFormatter.ofPattern( "yyyyMMdd'_'HHmmss.SSS" ) );
		//
		String fullFileName =
			"Z" +
			TimeS + "_" +
			base_FileCode +
			"_" +
			part_FileCode +
			".txt";
		//
		sis_File = new File( fileDrcS, fullFileName );
		//
		sis_File.setReadable( true );
		//
		if ( sis_File.exists() ) {
			if ( ShowBriefInfo || ShowFullInfo ) {
		    	out.format( "" );
		    	out.format( BTools.getMtLvESS( mtLv ) );
		    	out.format( BTools.getMtLvISS() );
		    	out.format( "delete File; " );
				out.format( "%s", BTools.getSLcDtTm() );
				out.format( "%n" );
			}
			sis_File.delete();
		}
		//
	    try {
	    	sis_File.createNewFile();
    	}
    	catch ( Exception Exc ) {
		//	Exc.printStackTrace( Err_PS );
	    	out.format( "===" );
			out.format( methodName + ": " );
			out.format( "create New File error !!! " );
			out.format( "Exception: %s; ", Exc.getMessage() );
			out.format( "%s", BTools.getSLcDtTm() );
			out.format( "%n" );
	    	out.format( "===" );
	    	out.format( BTools.getMtLvISS() );
			out.format( "fileDrcS: " + fileDrcS + "; " );
			out.format( "fullFileName: " + fullFileName + "; " );
			out.format( "%n" );
	    	//
			return;
	    }
	    //
	    if ( ShowFullInfo ) {
	    	out.format( "" );
	    	out.format( BTools.getMtLvESS( mtLv ) );
	    	out.format( BTools.getMtLvISS() );
			out.format( "fullFileName: " + fullFileName + "; " );
			out.format( "%s", BTools.getSLcDtTm() );
			out.format( "%n" );
	    }
		//
	    try {
	    	sis_Writer = new BufferedWriter( new FileWriter( sis_File ) );
    	}
    	catch ( Exception Exc ) {
	    	out.format( "===" );
			out.format( methodName + ": " );
			out.format( "create New Writer: " );
			out.format( "Exception: %s; ", Exc.getMessage() );
			out.format( "%s", BTools.getSLcDtTm() );
			out.format( "%n" );
	    	//
    		return ;
	    }
		//
	    wasOpenedFile = true;
	    //
	    if ( ShowFullInfo ) {
			OInfo = "";
			OInfo += BTools.getMtLvESS( mtLv );
			OInfo += methodName + ": ";
			OInfo += "fullFileName: " + fullFileName + "; ";
			out.format( "%s", BTools.getSLcDtTm() );
			info( OInfo );
	    }
	    //
	}
	
	
	public void info( String OInfo ) {
		//
		String methodName = modulCode + "." + "info";
		//
		out.format( "%s%n", OInfo );
		//
		String FOInfo = getFullInfoString( OInfo );
		//
		if ( !isFileOpen( methodName ) ) return;
		//
		outFile( FOInfo );
		//
        flushFile();
		//
	}
	
	public String getFullInfoString( String OInfo ) {
		//
		String Result = "";
		//
    	LocalDateTime LDT = LocalDateTime.now();
    	//
    	String TimeS = LDT.format( DateTimeFormatter.ofPattern( "yyyy.MM.dd HH:mm:ss.SSS" ) );
		//
		Result =
			TimeS +
		 	": " +
			OInfo +
			"\r\n" +
			"";
		//
		return Result;
	}
	
	private boolean isFileOpen( String SourceMethodName ) {
		//
		if ( !wasOpenedFile ) return false; 
		if ( !wasClosedFile ) return true; 
		//
		String methodName = modulCode + "." + "isFileOpen";
		//
		closedFileInfoCount ++;
		if ( closedFileInfoCount <= 3 ) {
	    	out.format( "===" );
//			out.format( methodName + ": " );
			out.format( methodName + "(from " + SourceMethodName + "): " );
	    	out.format( "File is closed !!!; " );
			out.format( "%s", BTools.getSLcDtTm() );
			out.format( "%n" );
		}
		//
		return false;
	}
	
	public void outFile( String FOInfo ) {
		//
		String methodName = modulCode + "." + "outFile";
		//
        try {
        	sis_Writer.write( FOInfo );
        }
        catch ( Exception Exc ) {
    		if ( writerErrorInfoCount < 2 ) {
    			writerErrorInfoCount ++;
        		out.format( "===" );
    			out.format( methodName + ": " );
    			out.format( "Writer.write error !!!; " );
    			out.format( "Exception: %s; ", Exc.getMessage() );
    			out.format( "%s", BTools.getSLcDtTm() );
    			out.format( "%n" );
    		}
			//
        }
		//
	}
	
	public void flushFile() {
		//
		String methodName = modulCode + "." + "flushFile";
		//
		try {
			sis_Writer.flush();
		}
		catch ( Exception Exc ) {
	    	out.format( "===" );
			out.format( methodName + ": " );
			out.format( "Writer.flush error !!!; " );
			out.format( "Exception: %s; ", Exc.getMessage() );
			out.format( "%s", BTools.getSLcDtTm() );
			out.format( "%n" );
		}
		//
	}
	
	
	
	public void onStop( int mtLv ) {
		//
		mtLv ++;
		//
		String OInfo = "";
		//
		String methodName = modulCode + "." + "onStop";
		//
		OInfo = "";
		OInfo += BTools.getMtLvESS( mtLv );
		OInfo += methodName + ": ";
		out.format( "%s", BTools.getSLcDtTm() );
		info( OInfo );
		//
		closeFile();
		//
	}
	
	
	private void closeFile() {
		//
		String methodName = modulCode + "." + "closeFile";
		//
		flushFile();
		//
		try {
			sis_Writer.close();
		}
		catch ( Exception Exc ) {
	    	out.format( "===" );
			out.format( methodName + ": " );
			out.format( "Writer.close error !!!; " );
			out.format( "Exception: %s; ", Exc.getMessage() );
			out.format( "%s", BTools.getSLcDtTm() );
			out.format( "%n" );
		}
		//
		wasClosedFile = true;
		//
	}
	
	
	
	
	
	
}