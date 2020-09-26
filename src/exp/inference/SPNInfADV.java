/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package exp.inference;

/**
 *
 * @author andreas
 */

import spn.GraphSPN;
import util.Parameter;
import data.Dataset;
import data.Partition;
import data.AdvQueryDataset;
import exp.RunSLSPN;
import java.util.ArrayList;
import java.util.Iterator;

public class SPNInfADV{

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		RunSLSPN.parseParameters(args);
		String prefix = "data/";
		Dataset d = new AdvQueryDataset(prefix+RunSLSPN.queryfile, prefix+RunSLSPN.evidencefile);

		GraphSPN spn = GraphSPN.load(Parameter.filename, d);//pass query to spn?
                //update leaf nodes
                
                spn.updatePolyNodes(d.getQueryValues());
                
		ArrayList<Double> llList = new ArrayList();
                ArrayList<Integer> queryIndex = ((AdvQueryDataset)d).getQueryIndexList();
                
                for(Integer  i: queryIndex){
                    System.out.println("queryIndex["+i+"]");
                }
                
                //double Xquery = 58; //percentage of query
                
                //double normLL = 0, normLLsq = 0;
		int attr = 0;
                int newattr = -1;
                double LL = 0, LLsq = 0; 
                double lastLL = 1;
		long tic = System.nanoTime();//currentTimeMillis(); //nanoTime()
                
                
                
		for(int inst=0; inst<d.getNumTesting(); inst++){
			double ill = 0;
                        double nll = 0;
                        attr = queryIndex.get(0);
                        System.out.println("queryIndex->attr: " +  attr);
                        queryIndex.remove(0);
			// P(q | ev) = P(ev,q) / P(e)
			((AdvQueryDataset) d).showJoint(inst, Partition.Testing);
			ill += spn.upwardAdvPass();
			((AdvQueryDataset) d).showEvidence(inst, Partition.Testing);
			ill -= spn.upwardAdvPass();
			System.out.println("first ill: " + ill);
                        if(attr != newattr){ //not first query val or different variables  
                            System.out.println("ill:"+ ill);
                            if(inst == 0){//first case
                                LL+=ill;
                            }else{
                                llList.add(LL);
                                LL = ill;
                            }
                            
                        }else{//
                            System.out.println("We're adding");
                            LL += ill;
                        }    
                        newattr = attr;
                        //nll = ill/Xquery;
                                               //normLL += nll;
                        //normLLsq += nll*nll;
     
		}
                if(LL != 0){
                    llList.add(LL);//last case
                }
                for (Double atom : llList) {
                    System.out.println("atom ll: " + atom);
                    lastLL *= Math.abs(atom);
                }
		lastLL*=-1;///keep log negattive	
                //LLsq = lastLL*ill;
		long toc = System.nanoTime();//currentTimeMillis(); //nanoTime()
                //System.out.println("SUM = " + LL);
		//LL /= d.getNumTesting();
		//LLsq /= d.getNumTesting();
		
                //normLL /= d.getNumTesting();
                //normLLsq /= d.getNumTesting();
                
                //double queryLL = LL/Xquery; 
                
                
		System.out.println("avg = "+lastLL);//+" +/- "+Math.sqrt(LLsq - LL*LL));
		System.out.println("Total time: "+(1.0*(toc-tic)/1000)+"s");
                System.out.println("size of list: " + d.getNumTesting());
                //System.out.println("normalized avg = " + normLL + " +/- " + Math.sqrt(normLLsq - normLL*normLL));
                
               // System.out.println("norm avg divided at end =" + queryLL);
		// avg = -21.734815 +/- 0.363440
		// Total time: 424.504467s

	}

}


