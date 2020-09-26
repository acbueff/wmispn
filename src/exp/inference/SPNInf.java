package exp.inference;

import spn.GraphSPN;
import util.Parameter;
import data.Dataset;
import data.Partition;
import data.QueryDataset;
import exp.RunSLSPN;
import java.util.ArrayList;

public class SPNInf {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		RunSLSPN.parseParameters(args);
		String prefix = "data/";
		Dataset d = new QueryDataset(prefix+RunSLSPN.queryfile, prefix+RunSLSPN.evidencefile);
		
		GraphSPN spn = GraphSPN.load(Parameter.filename, d);
                
		ArrayList<Double> llList = new ArrayList();
                
                
                double Xquery = 58; //percentage of query
                
                double normLL = 0, normLLsq = 0;
		
                double LL = 0, LLsq = 0;
		long tic = System.currentTimeMillis();
		for(int inst=0; inst<d.getNumTesting(); inst++){
			double ill = 0;
                        double nll = 0;
			// P(q | ev) = P(ev,q) / P(e)
			((QueryDataset) d).showJoint(inst, Partition.Testing);
			ill += spn.upwardPass();
			((QueryDataset) d).showEvidence(inst, Partition.Testing);
			ill -= spn.upwardPass();
			
			System.out.println(ill);
                        llList.add(ill);
			LL += ill;
			LLsq += ill*ill;
                        
                        
                        nll = ill/Xquery;
                        
                        normLL += nll;
                        normLLsq += nll*nll;
                        
                        
		}
		long toc = System.currentTimeMillis();
                System.out.println("SUM = " + LL);
		LL /= d.getNumTesting();
		LLsq /= d.getNumTesting();
		
                normLL /= d.getNumTesting();
                normLLsq /= d.getNumTesting();
                
                double queryLL = LL/Xquery; 
                
                
		System.out.println("avg = "+LL+" +/- "+Math.sqrt(LLsq - LL*LL));
		System.out.println("Total time: "+(1.0*(toc-tic)/1000)+"s");
                System.out.println("size of list: " + d.getNumTesting());
                System.out.println("normalized avg = " + normLL + " +/- " + Math.sqrt(normLLsq - normLL*normLL));
                
                System.out.println("norm avg divided at end =" + queryLL);
		// avg = -21.734815 +/- 0.363440
		// Total time: 424.504467s

	}

}

