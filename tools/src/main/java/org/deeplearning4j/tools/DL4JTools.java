package org.deeplearning4j.tools;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;


/**
 * shows content of some classes
 *
 * 
 *
 * @author clavvis 
 */

public class DL4JTools {
	//
	private SIS sis;
	//
	DL4JTools(
			SIS sis,
			String superiorModulCode
			) {
		//
		this.sis = sis;
		//
		initValues( superiorModulCode );
	}
	//
	private final String  baseModulCode = "AIT";
	private String  modulCode     = "";
	//
	
	private void initValues( String superiorModulCode ) {
		//
		modulCode = superiorModulCode + "." + baseModulCode;
		//
	}
	
	/**
	 * shows content of DataSet
	 * @param mtLv - method level
	 * @param itemCode - item = DataSet
	 * @param ds - DataSet
	 * @param in_Digits - input digits
	 * @param ot_Digits - output digits
	 * @param r_End_I - rows end index
	 * @param c_End_I - columns end index
	 */
	
	public void showDataSet(
			int mtLv,
			String itemCode,
			DataSet ds,
			int in_Digits,
			int ot_Digits,
			int r_End_I,
			int c_End_I
			) {
		//
        mtLv++;
		//
		String oinfo = "";
		//
		String methodName = modulCode + "." + "showDataSet";
		//
		if ( ds == null ) {
			oinfo = "";
			oinfo += BTools.getMtLvESS( mtLv );
			oinfo += methodName + ": ";
			oinfo += "\"" + itemCode + "\": ";
			oinfo += " == null !!!; ";
			oinfo += BTools.getSLcDtTm();
			sis.info( oinfo );
			return;
		}
		//
		oinfo = "";
		oinfo += BTools.getMtLvESS( mtLv );
		oinfo += methodName + ": ";
		oinfo += "\"" + itemCode + "\": ";
		oinfo += "in_Digits: " + in_Digits + "; ";
		oinfo += "ot_Digits: " + ot_Digits + "; ";
		sis.info( oinfo );
		oinfo = "";
		oinfo += BTools.getMtLvESS( mtLv );
		oinfo += BTools.getMtLvISS();
		oinfo += "r_End_I: " + r_End_I + "; ";
		oinfo += "c_End_I: " + c_End_I + "; ";
		oinfo += BTools.getSLcDtTm();
		sis.info( oinfo );
		oinfo = "";
		oinfo += BTools.getMtLvESS( mtLv );
		oinfo += BTools.getMtLvISS();
		oinfo += "ds: ";
		oinfo += ".numInputs: " + ds.numInputs() + "; ";
		oinfo += ".numOutcomes: " + ds.numOutcomes() + "; ";
		oinfo += ".numExamples: " + ds.numExamples() + "; ";
		oinfo += ".hasMaskArrays: " + BTools.getSBln( ds.hasMaskArrays() ) + "; ";
		sis.info( oinfo );
		//
		if ( in_Digits < 0 ) in_Digits = 0;
		if ( ot_Digits < 0 ) ot_Digits = 0;
		//
		INDArray in_INDA; // I = Input
		INDArray ot_INDA; // O = Output
		//
		in_INDA = ds.getFeatures();
		ot_INDA = ds.getLabels();
		//
		oinfo = "";
		oinfo += BTools.getMtLvESS( mtLv );
		oinfo += BTools.getMtLvISS();
		oinfo += "in_INDA: ";
		oinfo += ".rows: " + in_INDA.rows() + "; ";
		oinfo += ".columns: " + in_INDA.columns() + "; ";
		oinfo += ".rank: " + in_INDA.rank() + "; ";
		oinfo += ".shape: " + BTools.getSIntA( in_INDA.shape() ) + "; ";
		oinfo += ".length: " + in_INDA.length() + "; ";
		oinfo += ".size( 0 ): " + in_INDA.size( 0 ) + "; ";
		sis.info( oinfo );
		//
		if ( ot_INDA != null ) {
			oinfo = "";
			oinfo += BTools.getMtLvESS( mtLv );
			oinfo += BTools.getMtLvISS();
			oinfo += "ot_INDA: ";
			oinfo += ".rows: " + ot_INDA.rows() + "; ";
			oinfo += ".columns: " + ot_INDA.columns() + "; ";
			oinfo += ".rank: " + ot_INDA.rank() + "; ";
			oinfo += ".shape: " + BTools.getSIntA( ot_INDA.shape() ) + "; ";
			oinfo += ".length: " + ot_INDA.length() + "; ";
			oinfo += ".size( 0 ): " + ot_INDA.size( 0 ) + "; ";
			sis.info( oinfo );
		} else {
			oinfo = "";
			oinfo += BTools.getMtLvESS( mtLv );
			oinfo += BTools.getMtLvISS();
			oinfo += "ot_INDA == null ! ";
			sis.info( oinfo );
		}
		//
		if ( in_INDA.rows() != ot_INDA.rows() ) {
			oinfo = "===";
			oinfo += methodName + ": ";
			oinfo += "in_INDA.rows() != ot_INDA.rows() !!! ; ";
			oinfo += BTools.getSLcDtTm();
			sis.info( oinfo );
			//
			return;
		}
		//
		boolean wasShownTitle = false;
		//
		InfoLine il;
		InfoValues iv;
		//
		double j_Dbl = -1;
		//
		int i_CharsCount = BTools.getIndexCharsCount( in_INDA.rows() - 1 );
		//
		oinfo = "";
		oinfo += BTools.getMtLvESS( mtLv );
		oinfo += BTools.getMtLvISS();
		oinfo += "Data: j: IN->I0; ";
		sis.info( oinfo );
		//
		for ( int i = 0; i < in_INDA.rows(); i++ ) {
			//
			if ( i > r_End_I ) break;
			//
			il = new InfoLine();
			//
			iv = new InfoValues( "i", "" ); il.ivL.add( iv );
			iv.vsL.add( BTools.getSInt( i, i_CharsCount ) );
			//
			iv = new InfoValues( "", "", "" ); il.ivL.add( iv );
			iv.vsL.add( "" );
			//
			int c_I = 0;
			//
			for ( int j = in_INDA.columns() - 1; j >= 0; j-- ) {
				//
				if ( c_I > c_End_I ) break;
				//
				j_Dbl = in_INDA.getDouble( i, j );
				//
				iv = new InfoValues( "In", "j", BTools.getSInt( j ) ); il.ivL.add( iv );
				iv.vsL.add( BTools.getSDbl( j_Dbl, in_Digits, true, in_Digits + 4 ) );
				//
				c_I++;
			}
			//
			iv = new InfoValues( "", "", "" ); il.ivL.add( iv );
			iv.vsL.add( "" );
			//
			c_I = 0;
			//
			if ( ot_INDA != null ) {
				for ( int j = ot_INDA.columns() - 1; j >= 0; j-- ) {
					//
					if ( c_I > c_End_I ) break;
					//
					j_Dbl = ot_INDA.getDouble( i, j );
					//
					iv = new InfoValues( "Ot", "j", BTools.getSInt( j ) ); il.ivL.add( iv );
					iv.vsL.add( BTools.getSDbl( j_Dbl, ot_Digits, true, ot_Digits + 4 ) );
					//
					c_I++;
				}
			}
			//
			if ( !wasShownTitle ) {
			    oinfo = il.getTitleLine( mtLv, 0 ); sis.info( oinfo );
			    oinfo = il.getTitleLine( mtLv, 1 ); sis.info( oinfo );
			    oinfo = il.getTitleLine( mtLv, 2 ); sis.info( oinfo );
//			    oinfo = il.getTitleLine( mtLv, 3 ); sis.info( oinfo );
//			    oinfo = il.getTitleLine( mtLv, 4 ); sis.info( oinfo );
				wasShownTitle = true;
			}
			oinfo = il.getValuesLine( mtLv ); sis.info( oinfo );
		}
		//
	}
	
	/**
	 * shows content of INDArray
	 * not turned shows first rows and than columns
	 * @param mtLv - method level
	 * @param itemCode
	 * @param INDA - INDArray
	 * @param digits
	 * @param r_End_I - rows end index
	 * @param c_End_I - clolumns end index
	 */
	public void showINDArray(
			int mtLv,
			String itemCode,
			INDArray INDA,
			int digits,
			int r_End_I,
			int c_End_I
			) {
		//
		showINDArray( mtLv, itemCode, INDA, digits, r_End_I, c_End_I, false );
	}
	
	/**
	 * shows content of INDArray
	 * @param mtLv - method level
	 * @param itemCode
	 * @param INDA - INDArray
	 * @param digits
	 * @param r_End_I - rows end index
	 * @param c_End_I - clolumns end index
	 * @param Turned - not turned show first rows and than columns
	 */
	public void showINDArray(
			int mtLv,
			String itemCode,
			INDArray INDA,
			int digits,
			int r_End_I,
			int c_End_I,
			boolean Turned
			) {
		//
        mtLv++;
		//
		String oinfo = "";
		//
		String methodName = modulCode + "." + "showINDArray";
		//
		if ( INDA == null ) {
			oinfo = "";
			oinfo += BTools.getMtLvESS( mtLv );
			oinfo += methodName + ": ";
			oinfo += "\"" + itemCode + "\": ";
			oinfo += " == null !!!; ";
			oinfo += BTools.getSLcDtTm();
			sis.info( oinfo );
			return;
		}
		//
		oinfo = "";
		oinfo += BTools.getMtLvESS( mtLv );
		oinfo += methodName + ": ";
		oinfo += "\"" + itemCode + "\": ";
		oinfo += "digits: " + digits + "; ";
		oinfo += "r_End_I: " + r_End_I + "; ";
		oinfo += "c_End_I: " + c_End_I + "; ";
		oinfo += "Turned: " + Turned + "; ";
		oinfo += BTools.getSLcDtTm();
		sis.info( oinfo );
		//
		if ( digits < 0 ) digits = 0;
		//
		oinfo = "";
		oinfo += BTools.getMtLvESS( mtLv );
		oinfo += BTools.getMtLvISS();
		oinfo += "rows: " + INDA.rows() + "; ";
		oinfo += "columns: " + INDA.columns() + "; ";
		oinfo += "rank: " + INDA.rank() + "; ";
		oinfo += "shape: " + BTools.getSIntA( INDA.shape() ) + "; ";
		oinfo += "length: " + INDA.length() + "; ";
		oinfo += "size( 0 ): " + INDA.size( 0 ) + "; ";
		sis.info( oinfo );
		//
		boolean wasShownTitle = false;
		//
		InfoLine il;
		InfoValues iv;
		//
		double j_Dbl = -1;
		//
		int i_CharsCount = BTools.getIndexCharsCount( INDA.rows() - 1 );
		//
		if ( !Turned ) { //= standard
			oinfo = "";
			oinfo += BTools.getMtLvESS( mtLv );
			oinfo += BTools.getMtLvISS();
			oinfo += "Data: j: IN->I0; ";
			sis.info( oinfo );
			//
			for ( int i = 0; i < INDA.rows(); i++ ) {
				//
				if ( i > r_End_I ) break;
				//
				il = new InfoLine();
				//
				iv = new InfoValues( "i", "" ); il.ivL.add( iv );
				iv.vsL.add( BTools.getSInt( i, i_CharsCount ) );
				//
				int c_I = 0;
				//
				for ( int j = INDA.columns() - 1; j >= 0; j-- ) {
					//
					if ( c_I > c_End_I ) break;
					//
					j_Dbl = INDA.getDouble( i, j );
					//
					iv = new InfoValues( "j", "", BTools.getSInt( j ) ); il.ivL.add( iv );
					iv.vsL.add( BTools.getSDbl( j_Dbl, digits, true, digits + 4 ) );
					//
					c_I++;
				}
				//
				if ( !wasShownTitle ) {
				    oinfo = il.getTitleLine( mtLv, 0 ); sis.info( oinfo );
				    oinfo = il.getTitleLine( mtLv, 1 ); sis.info( oinfo );
				    oinfo = il.getTitleLine( mtLv, 2 ); sis.info( oinfo );
//				    oinfo = il.getTitleLine( mtLv, 3 ); sis.info( oinfo );
//				    oinfo = il.getTitleLine( mtLv, 4 ); sis.info( oinfo );
					wasShownTitle = true;
				}
				oinfo = il.getValuesLine( mtLv ); sis.info( oinfo );
			}
		}
		else { // = turned
			oinfo = "";
			oinfo += BTools.getMtLvESS( mtLv );
			oinfo += BTools.getMtLvISS();
			oinfo += "Data: ";
			sis.info( oinfo );
			//
			for ( int i = 0; i < INDA.columns(); i++ ) {
				//
				if ( i > c_End_I ) break;
				//
				il = new InfoLine();
				//
				iv = new InfoValues( "i", "" ); il.ivL.add( iv );
				iv.vsL.add( BTools.getSInt( i, i_CharsCount ) );
				//
				int r_I = 0;
				//
				for ( int j = 0; j < INDA.rows(); j++ ) {
					//
					if ( r_I > r_End_I ) break;
					//
					j_Dbl = INDA.getDouble( j, i );
					//
					iv = new InfoValues( "j", "", BTools.getSInt( j ) ); il.ivL.add( iv );
					iv.vsL.add( BTools.getSDbl( j_Dbl, digits, true, digits + 4 ) );
					//
					r_I++;
				}
				//
				if ( !wasShownTitle ) {
				    oinfo = il.getTitleLine( mtLv, 0 ); sis.info( oinfo );
				    oinfo = il.getTitleLine( mtLv, 1 ); sis.info( oinfo );
				    oinfo = il.getTitleLine( mtLv, 2 ); sis.info( oinfo );
//				    oinfo = il.getTitleLine( mtLv, 3 ); sis.info( oinfo );
//				    oinfo = il.getTitleLine( mtLv, 4 ); sis.info( oinfo );
					wasShownTitle = true;
				}
				oinfo = il.getValuesLine( mtLv ); sis.info( oinfo );
			}
		}
		//
	}
	
	
	
	
}