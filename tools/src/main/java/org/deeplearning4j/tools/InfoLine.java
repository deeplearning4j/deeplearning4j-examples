package org.deeplearning4j.tools;

import java.util.ArrayList;
import java.util.List;


/**
 * save values and titles for one line
 *
 * 
 *
 * @author clavvis 
 */

public class InfoLine {
	//
	public InfoLine() {
		//
	}
	//
	public List< InfoValues > ivL = new ArrayList< InfoValues >();
	//
	
	public String getTitleLine( int mtLv, int title_I ) {
		//
		String info = "";
		//
		info = "";
		info += BTools.getMtLvESS( mtLv );
		info += BTools.getMtLvISS();
		info += "|";
		//
		InfoValues i_IV;
		//
		String i_ValuesS = "";
		//
		int i_VSLen = -1;
		//
		String i_TitleS = "";
		//
		for ( int i = 0; i < ivL.size(); i ++ ) {
			//
			i_IV = ivL.get( i );
			//
			i_ValuesS = i_IV.getValues();
			//
			i_VSLen = i_ValuesS.length();
			//
			i_TitleS = ( title_I < i_IV.titleA.length )? i_IV.titleA[ title_I ] : "";
			//
			i_TitleS = i_TitleS + BTools.getSpaces( i_VSLen );
			//
			info += i_TitleS.substring( 0, i_VSLen - 1 );
			//
			info += "|";
		}
		//
		return info;
	}
	
	public String getValuesLine( int mtLv ) {
		//
		String info = "";
		//
		info += BTools.getMtLvESS( mtLv );
		info += BTools.getMtLvISS();
		info += "|";
		//
		InfoValues i_IV;
		//
		for ( int i = 0; i < ivL.size(); i ++ ) {
			//
			i_IV = ivL.get( i );
			//
			info += i_IV.getValues();
		}
		//
		return info;
	}
	
	
		
}