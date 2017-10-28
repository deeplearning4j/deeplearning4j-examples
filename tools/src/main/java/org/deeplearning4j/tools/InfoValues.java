package org.deeplearning4j.tools;

import java.util.ArrayList;
import java.util.List;

/**
 * save value and titles
 * titles strings in array create one title column
 * 
 *
 * @author clavvis 
 */

public class InfoValues {
	//
	public InfoValues( String... titleA ) {
		//
		for ( int i = 0; i < this.titleA.length; i++ ) this.titleA[ i ] = "";
		//
		int Max_K = Math.min( this.titleA.length - 1, titleA.length - 1 );
		//
		for ( int i = 0; i <= Max_K; i++ ) this.titleA[ i ] = titleA[ i ];
		//
	}
	//
	String[] titleA = new String[ 6 ];
	//
	// VS = Values String
	public List< String > vsL = new ArrayList< String >();
	//
	//
	
	public String getValues() {
		//
		String info = "";
		//
		for ( int i = 0; i < vsL.size(); i ++ ) {
			//
			info += vsL.get( i ) + "|";
		}
		//
		return info;
	}
	
}