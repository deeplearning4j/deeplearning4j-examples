package org.deeplearning4j.tools;

import java.text.DecimalFormat;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * includes several base tools
 *
 * 
 *
 * @author clavvis 
 */

//B = Base
public class BTools {
	//
	
	/**
	 * 
	 * @param mtLv - method level
	 * @return method level external shift string
	 */
	public static String getMtLvESS( int mtLv ) {
		//  MtLvESS = Method Level External Shift String 
		//
		String Result = "";
		//
	//	String LvS = ". ";
		String LvS = ".";
		//
		for ( int K = 1; K <= mtLv; K ++ ) {
			//
			Result = Result + LvS;
		}
		//
		return Result;
	}
	
	/**
	 * 
	 * @return method level internal shift string
	 */
	public static String getMtLvISS() {
		//  MtLvISS = Method Level Intern Shift String 
		//
	//	String Result = "..";
	//	String Result = "~";
		String Result = " ";
		//
		return Result;
	}
	
	/**
	 * 
	 * @param SpacesCount
	 * @return spaces
	 */
	public static String getSpaces( int SpacesCount ) {
		//
		String Info = "";
		//
		for ( int K = 1; K <= SpacesCount; K ++ ) {
			Info += " ";
		}
		//
		//
		return Info;
	}
	
	/**
	 * 
	 * @param BlnA
	 * @return boolean(s) as string - true = T, false = F
	 */
	public static String getSBln( boolean... BlnA ) {
		//
		String Info = "";
		//
		if ( BlnA == null ) return "?";
		if ( BlnA.length == 0 ) return "?";
		//
		for ( int K = 0; K < BlnA.length; K ++ ) {
			//
			Info += ( BlnA[ K ] )? "T" : "F";
		}
		//
		return Info;
	}
		
	/**
	 * 
	 * @param Value - value
	 * @param DecPrec - decimal precision
	 * @param ShowPlusSign - show plus sign
	 * @param StringLength - string length
	 * @return double as string
	 */
	public static String getSDbl( double Value, int DecPrec, boolean ShowPlusSign, int StringLength ) {
		//
		String Info = "";
		//
		String SDbl = getSDbl( Value, DecPrec, ShowPlusSign );
		//
		if ( SDbl.length() >= StringLength ) return SDbl;
		//
//		String SpacesS = "            ";
		String SpacesS = getSpaces( StringLength );
		//
		Info = SpacesS.substring( 0, StringLength - SDbl.length() ) + SDbl;
		//
		return Info;
	}
	
	/**
	 * 
	 * @param Value - value
	 * @param DecPrec - decimal precision
	 * @param ShowPlusSign - show plus sign
	 * @return double as string
	 */
	public static String getSDbl( double Value, int DecPrec, boolean ShowPlusSign ) {
		//
		String PlusSign = "";
		//
		if ( ShowPlusSign && Value  > 0 ) PlusSign = "+";
		if ( ShowPlusSign && Value == 0 ) PlusSign = " ";
		//
		return PlusSign + getSDbl( Value, DecPrec );
	}
	
	/**
	 * 
	 * @param Value - value
	 * @param DecPrec - decimal precision
	 * @return double as string
	 */
	public static String getSDbl( double Value, int DecPrec ) {
		//
		String Result = "";
		//
		if ( Double.isNaN( Value ) ) return "NaN";
		//
		if ( DecPrec < 0 ) DecPrec = 0;
		//
		DecimalFormat DcmFrm; // = getPriceDecimalFormat( DPCPlus + 1 - PipScale );
		//
		String DFS = "0";
		//
		if ( DecPrec == 0 ) {
			DFS = "0";
		}
		else {
			int idx = 0;
			DFS = "0.";
			while ( idx < DecPrec ) {
				DFS = DFS + "0";
				idx ++;
				if ( idx > 100 ) break;
			}
		}
		//
		DcmFrm = new DecimalFormat( DFS );
		//
		if ( Value != 0 ) {
			Result = DcmFrm.format( Value );
		}
		else {
			Result = DFS;
		}
		//
		return Result;
	}
	
	/**
	 * 
	 * @param Value
	 * @param CharsCount - chars count
	 * @return int as string
	 */
	public static String getSInt( int Value, int CharsCount ) {
		//
		return getSInt( Value, CharsCount, ' ' );
	}
	
	/**
	 * 
	 * @param Value
	 * @param CharsCount - chars count
	 * @param LeadingChar - leading char
	 * @return int as string
	 */
	public static String getSInt( int Value, int CharsCount, char LeadingChar ) {
		//
		String Result = "";
		//
		if ( CharsCount <= 0 ) {
			return getSInt( Value );
		}
		//
		String FormatS = "";
		if ( LeadingChar == '0' ) {
			FormatS = "%" + LeadingChar + Integer.toString( CharsCount ) + "d";
		}
		else {
			FormatS = "%" + Integer.toString( CharsCount ) + "d";
		}
		//
		Result = String.format( FormatS, Value );
		//
		return Result;
	}
	
	/**
	 * 
	 * @param Value
	 * @return int as string
	 */
	public static String getSInt( int Value ) {
		//
		String Result = "";
		//
		Result = String.format( "%d", Value );
		//
		return Result;
	}
	
	/**
	 * 
	 * @param IntA
	 * @return int[] as string
	 */
	public static String getSIntA( int[] IntA ) {
		//
		String Info = "";
		//
		if ( IntA == null ) return "?";
		if ( IntA.length == 0 ) return "?";
		//
		for ( int K = 0; K < IntA.length; K ++ ) {
			//
            Info += ( Info.isEmpty() )? "" : ", ";
			Info += BTools.getSInt( IntA[ K ] );
		}
		//
		return Info;
	}
	
	/**
	 * 
	 * @param MaxIndex
	 * @return chars count for index (max value)
	 */
	public static int getIndexCharsCount( int MaxIndex ) {
		//
		int CharsCount = 1;
		//
		if ( MaxIndex <= 0 ) return 1;
		//
		CharsCount = (int)Math.log10( MaxIndex ) + 1;
		//
		return CharsCount;
	}
	
	/**
	 * 
	 * @return lokal datetime as string (format "mm:ss.SSS" )
	 */
	public static String getSLcDtTm() {
		//
		return getSLcDtTm( "mm:ss.SSS" );
	}
	
	/**
	 * 
	 * @param FormatS string format
	 * @return lokal datetime as string
	 */
	public static String getSLcDtTm( String FormatS ) {
		//
		String Result = "?";
		//
    	LocalDateTime LDT = LocalDateTime.now();
    	//
    	Result = "LDTm: " +  LDT.format( DateTimeFormatter.ofPattern( FormatS ) );
    	//
		return Result;
	}
	
	
	
	
	
	
}