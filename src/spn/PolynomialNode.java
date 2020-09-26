/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package spn;

/**
 *
 * @author andreas
 */
import data.Dataset;
import data.Partition;
import java.net.URLClassLoader;
import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.analysis.integration.SimpsonIntegrator;

import java.util.ArrayList;
import java.util.Arrays;

public class PolynomialNode extends Node implements DatasetDep {
	private static final long serialVersionUID = 1L;
	public static double smooth = 0.0000001;
	private double logvals[];
        private double querylogvals[];
	private final int attr;//this is the index in the overall array
        private final int feature; //origional feature this array corresponds to
        private final int featureIndex; //index of feature 
        private final double min; //min value for the feature range
        private final double max; //max value for the feature range
	private Dataset d;
	private final int[] instances;
        private final double pdf;
        private final ArrayList<Double> polynomials;
        private final ArrayList<ArrayList<Double>> allpoly;
        private final double[] p_counts;
        private double p_countsum;
        private final int vartype;
        
        //Query related
        public double query_value =0;
	
	public PolynomialNode(Dataset d, int variable, int instances[]) {
		this.d = d;
		attr = variable;
		this.instances = instances; 
                pdf = d.getPdfUnivariate()[attr];
                feature = d.getOldIndecies()[attr];
                featureIndex = d.getFeatureIndecies()[attr];
                min = d.getMinMax()[0][attr];
                max = d.getMinMax()[1][attr];
                vartype = d.getVariableType()[attr];
                //System.out.println("Polynomial Node " + feature);
                //System.out.println("attr: " + attr);
                //System.out.println("old index: " + feature);
                //System.out.println("feature index: " + featureIndex);
                //System.out.println("min("+min+") " + "max("+max+")");
                //System.out.println("pfd val: " + pdf);
                //polynomials = null;
                
                
                allpoly = new ArrayList<ArrayList<Double>>(d.getPolyList());
                
                   //print contents 
                   /*
                     System.out.println("all polyList: " );   
                    for(ArrayList<Double> innerLs : allpoly) {
                        for(double e : innerLs){
                            System.out.print(" " + e + " ");
                        }
                        System.out.println();
                    }*/
                
                int i = 0;
                int index = 0;
                
                for(ArrayList<Double> poly : allpoly){
                    if(poly.get(0) == (double)attr){
                        
                        index = i;
                    }
                    i++;
                }
                
                polynomials = (ArrayList<Double>)allpoly.get(index).clone();//ignore the first element
		polynomials.remove(0);
                p_counts = new double[2];
		p_countsum=2*smooth;
                
                for(int val=0; val<2; val++){
			p_counts[val] = smooth;
		}
                for(int inst : instances){
			d.show(inst, Partition.Training);
			int setting = d.getValues()[attr];
			p_counts[setting]++;
			p_countsum++;
		}
                
                
		reset();
	}
	
	public void setDataset(Dataset d) {
		this.d = d;
	}
	
	public void reset(){
		int numvals = d.getAttrSizes()[attr]; //should be 2
		double counts[] = new double[numvals];
		double countsum=numvals*smooth; // should be 0.0000002
		
		for(int val=0; val<numvals; val++){
			counts[val] = smooth;
		}
		
		for(int inst : instances){
			d.show(inst, Partition.Training);
			int setting = d.getValues()[attr];
			counts[setting]++;
			countsum++;
		}
		
		logvals = new double[numvals];
                double probOne = pdf;
                double probZero = (1- pdf);
                double logsum = 0;
                   
		for(int val=0; val<numvals; val++){
                        
                        if(val == 0){
                            logvals[val] = (Math.log(counts[val]) - Math.log(countsum)) + Math.log(probZero);
                        }else{
                            logvals[val] = (Math.log(counts[val]) - Math.log(countsum)) + Math.log(probOne);
                        }
			
                        logsum += logvals[val];
		}
                //normalize it based on new probablities
                for(int log = 0; log < numvals; log++){
                    
                    logvals[log] = -1*(logvals[log] / logsum);
                }
               //System.out.println("logvals["+0+"]: " +logvals[0]);
               //System.out.println("logvals["+1+"]: " +logvals[1]);
                
                
	}
	
	@Override
	public void eval() {
		int setting = d.getValues()[attr];//
		double new_logval_;
		
		if(setting == -1){
			new_logval_ = 0;
		} else {
			new_logval_ = logvals[setting];
		}
		
		if(new_logval_ == logval_){
			dirty = false;
		} else {
			logval_ = new_logval_;
			dirty = true;
		}
		
	}
        
        
        public void queryReset(){
            System.out.println("query for range of values a < x <=b");
        }
        
        public void updateNewLog(double[] queryval){
            
            System.out.println("Polynomial Node " + feature);
                System.out.println("attr: " + attr);
                System.out.println("old index: " + feature);
                System.out.println("feature index: " + featureIndex);
                System.out.println("min("+min+") " + "max("+max+")");
               System.out.println("pfd val: " + pdf);
            
            
            //add smoothing like in eval()
            int numvals = 2;//d.getAttrSizes()[attr]; //should be 2
            double q_val = queryval[attr];
            System.out.println("q_val: " + q_val);
            double counts[] = new double[numvals];
            double countsum=numvals*smooth; // should be 0.0000002
            querylogvals = new double[numvals];
            for(int val=0; val<numvals; val++){
			counts[val] = smooth;
            }
            
            
            ///need to make this more like reset(), COUNT 1's n 0's!
            ArrayList list = new ArrayList(Arrays.asList(polynomials));
            if(q_val == 1 || q_val == 0){
                
                querylogvals = logvals;
                System.out.println("No update Polynode querylogvals[1]" + querylogvals[1]);
                System.out.println("No update Polynode querylogvals[0]" + querylogvals[0]);
            }else{//complex query
                
                double probOne;
                if(featureIndex != 1){ //integrate from val to max
                    System.out.println("Integrate " + q_val + " with max " + max);
                    System.out.println("polynomials ->" + polynomials);
                    probOne = integrateQuery(q_val, max, polynomials);
                }else{
                    
                    System.out.println("Or Integrate " + q_val + " with min" + min);
                    System.out.println("polynomials ->" + polynomials);
                    probOne = integrateQuery( min, Math.abs(q_val), polynomials);
                }
                double probZero = (1- probOne);
                double logsum = 0;
                System.out.println("Feature bin: " + attr);
                System.out.println("updateNewLog ->  probOne: " + probOne);
                 System.out.println("updateNewLog ->  probZero: " + probZero);
                System.out.println("updateNewLog ->  p_counts[1]: " + p_counts[1]);
                System.out.println("updateNewLog ->  p_counts[0]: " + p_counts[0]);
                System.out.println("updateNewLog ->  p_countsum: " + p_countsum);
                
                for(int val=0; val<numvals; val++){
                    if(val == 0){
                      
                        querylogvals[val] = (Math.log(p_counts[val]) - Math.log(p_countsum)) + Math.log(probZero);
                    }else{
                        querylogvals[val] = (Math.log(p_counts[val]) - Math.log(p_countsum)) + Math.log(probOne);
                    }
                    logsum += querylogvals[val];
                }
                    //normalize it based on new probablities
                for(int log = 0; log < numvals; log++){

                    querylogvals[log] = -1*(querylogvals[log] / logsum);
                    System.out.println("updateNewLog -> querylogvals["+log+"]" + querylogvals[log]);
                }
                
               
            }
            
        }
        
        public void evalAdv(){
                int setting = d.getValues()[attr];//
                //System.out.println("queryEval() -> setting: " +setting);
		double new_logval_;
		//add smoothing like in eval()
		if(setting == -1){
			new_logval_ = 0;
		} else {
			new_logval_ = querylogvals[setting];
		}
		//System.out.println("queryEval() -> new_logval_: " +new_logval_);
		if(new_logval_ == complex_logval_){
			dirty = false;
		} else {
			complex_logval_ = new_logval_;
			dirty = true;
		}
                //System.out.println("queryEval() -> complex_logval_: " +complex_logval_);
            
            
        }
        
        
        public  UnivariateFunction genFctn(ArrayList constants){
        UnivariateFunction f = new UnivariateFunction() {
            @Override
            public double value(double v) {
                double res=0;
                for (int i=0;i<constants.size();i++){
                    res+=((double)(constants.get(i))* java.lang.Math.pow(v,i));
                }
                return res;
            }
        };
        return f;

        }

        public double integrateQuery(double a,double b,ArrayList constants){
            SimpsonIntegrator sim =new SimpsonIntegrator();
            UnivariateFunction f = genFctn(constants);
            double integratedValue = sim.integrate(300,f,a,b);
            return integratedValue;

        }

        public void advQuery(String []args){
            ArrayList list = new ArrayList(Arrays.asList(polynomials));
            double result = integrateQuery(1.513,1.514,list);
            System.out.println(result);
        }

	
}

