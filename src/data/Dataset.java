package data;

import java.io.Serializable;
import java.util.ArrayList; 

public abstract class Dataset implements Serializable{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private int currentFold = 0;
	public int numFeats;
	
	public abstract double getNumFolds();

	public void setFold(int f) {
		currentFold = f;
	}
	
	public abstract int getNumClasses();

	public abstract int getNumTesting();
	public abstract int getNumTraining();
	public abstract int getNumValidation();
        public abstract double[] getQueryValues();
	public abstract int trueLabel();

	public void show(int i, Partition testing){
		show(i, testing, true);
	}
	
	public abstract void show(int i, Partition testing, boolean b);



	public int getNumItems(Partition p) {
		switch (p) {
		case Training: return getNumTraining(); 
		case Validation: return getNumValidation(); 
		case Testing: return getNumTesting();
		default:
			break;
		}
		return 0;
	}

	public abstract int getUniqueInstanceID();

	public abstract int getNumFeatures();
	public abstract int[] getAttrSizes();
	public abstract int[] getValues();
        public abstract double[] getPdfUnivariate();
        public abstract int[] getVariableType();
        public abstract int[] getOldIndecies();
        public abstract int[] getFeatureIndecies();
	public abstract double[][] getMinMax();
        public abstract ArrayList<ArrayList<Double>> getPolyList(); //ArrayList<ArrayList<Integer>> outer = new ArrayList<ArrayList<Integer>>();
	

}
