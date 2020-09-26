package exp;

import data.ContinuousDiscretized;
import data.PolynomialDiscretized;
import slalg.VarInstSplit;
import spn.GraphSPN;
import spn.Node;
import spn.SmoothedMultinomialNode;
import spn.PolynomialNode;
import util.Parameter;
import data.Dataset;
import data.Discretized;
import data.Partition;
import exp.inference.SPNInfPLL;
import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.analysis.integration.SimpsonIntegrator;

public class RunSLSPN {
	static Dataset d;
	public static int data_id;
        public static int bins_perFeature;
	public static String queryfile;
	public static String evidencefile;
        public static String outputfile;
	
	public static Class ds[] = new Class[] {Discretized.EachMovie.class,
			Discretized.MSWeb.class, 
			Discretized.KDD.class,
			Discretized.S20NG.class,
			Discretized.Abalone.class,
			Discretized.Adult.class,
			Discretized.Audio.class,
			Discretized.Book.class,
			Discretized.Covertype.class,
			Discretized.Jester.class,
			Discretized.MSNBC.class, // 10
			Discretized.Netflix.class,
			Discretized.NLTCS.class,
			Discretized.Plants.class,
			Discretized.R52.class,
			Discretized.School.class,
			Discretized.Traffic.class,
			Discretized.WebKB.class,
			Discretized.Wine.class, // 18
			Discretized.Accidents.class,
			Discretized.Ad.class,//20
			Discretized.BBC.class,
			Discretized.C20NG.class,
			Discretized.CWebKB.class,
			Discretized.DNA.class,
			Discretized.Kosarek.class,
			Discretized.Retail.class,
			Discretized.Pumsb_Star.class,
			Discretized.CR52.class,// 28
                        Discretized.BinaryWine.class,//29
                        ContinuousDiscretized.ContinuousWine.class,//30
                        ContinuousDiscretized.BigBinaryWine.class, //31
                        ContinuousDiscretized.EqualWidthBinWine.class,//32
                        Discretized.MdlpWine.class, //33 
                        ContinuousDiscretized.EwBinOne.class, //34
                        ContinuousDiscretized.EwCloud.class,//35
                        ContinuousDiscretized.EwStatlog.class, //36
                        ContinuousDiscretized.EwBoone.class, //37
                        PolynomialDiscretized.PolyCloud.class, //38
                        PolynomialDiscretized.PolyAustralian.class, //39
                        PolynomialDiscretized.PolyAnnealU.class, //40
                        PolynomialDiscretized.PolyAuto.class, //41
                        PolynomialDiscretized.PolyCars.class,//42
                        PolynomialDiscretized.PolyCleave.class, //43
                        PolynomialDiscretized.PolyCRX.class, //44
                        PolynomialDiscretized.PolyDiabetes.class, //45
                        PolynomialDiscretized.PolyGerman.class, //46
                        PolynomialDiscretized.PolyGermanOrg.class, //47
                        PolynomialDiscretized.PolyHeart.class, //48
                        PolynomialDiscretized.PolyIris.class, //49
                        PolynomialDiscretized.PolyStatlog.class,//50
                        PolynomialDiscretized.PolyBoone.class,//51
                        ContinuousDiscretized.EwAnnealU.class, //52
                        ContinuousDiscretized.EwAustralian.class, //53
                        ContinuousDiscretized.EwAuto.class, //54
                        ContinuousDiscretized.EwCars.class, //55
                        ContinuousDiscretized.EwCleave.class, //56
                        ContinuousDiscretized.EwCRX.class, //57
                        ContinuousDiscretized.EwDiabetes.class, //58
                        ContinuousDiscretized.EwGerman.class, //59
                        ContinuousDiscretized.EwGermanOrg.class, //60
                        ContinuousDiscretized.EwHeart.class, //61
                        ContinuousDiscretized.EwIris.class, //62
                        ContinuousDiscretized.EwBurkina.class, //63
                        PolynomialDiscretized.BetaFake.class, //64
                        PolynomialDiscretized.FakeGauss.class, //65
                        PolynomialDiscretized.FakeExpon.class, //66
                        PolynomialDiscretized.FakeMix.class //67
                        }; 
	
	/**
         * 
         * 
         * 
	 * @param args
	 */
	public static void main(String[] args) {
		long tic = System.currentTimeMillis();
		parseParameters(args);

		try {
			d = (Dataset) ds[data_id].newInstance();
		} catch (Exception e) {
			e.printStackTrace();
		}

		System.out.println("\nCP "+VarInstSplit.clusterPenalty+"\tGF "+VarInstSplit.gfactor+"\tIT "+VarInstSplit.indepInstThresh);

		// Run SL
		VarInstSplit slalg = new VarInstSplit();
               
                
                if(d instanceof PolynomialDiscretized){
                    System.out.println("Will begin learning SPN with polynomials in the univariate case");
                    
                    
                    GraphSPN spn = slalg.learnPolyStructure(d);
                    
                    long toc = System.currentTimeMillis();
		
                    double highestValidation = Double.NEGATIVE_INFINITY;

                    double smoo = 1.0;
                    for(smoo = 2.0; smoo>=0.02; smoo*=0.5){
                             //0.0000001
                            PolynomialNode.smooth = smoo;
                            for(Node n : spn.order){
                                    if(n instanceof PolynomialNode){
                                            //System.out.println("instance Polynomial");
                                            PolynomialNode smn = (PolynomialNode) n;
                                            smn.reset();
                                    }
                                    
                                    if(n instanceof SmoothedMultinomialNode){
                                        //System.out.println("instance Multinomial");
                                            SmoothedMultinomialNode smn = (SmoothedMultinomialNode) n;
                                            smn.reset();
                                    }

                            }
                            System.out.println("\nCP "+VarInstSplit.clusterPenalty+"\tGF "+VarInstSplit.gfactor+"\tIT "+VarInstSplit.indepInstThresh+"\tSmoo"+SmoothedMultinomialNode.smooth);

                            //				spn.printDepth(10);
                            double llh = spn.llh(Partition.Training);
                            System.out.println("Train LLH: "+llh);

                            llh = spn.llh(Partition.Validation);
                            System.out.println("Validation LLH: "+llh);
                            if(llh > highestValidation){
                                    highestValidation = llh;
                                    spn.write(util.Parameter.filename);
                                    System.out.println("***");
                            }
                    } // end smoothing loop
                    spn = GraphSPN.load(util.Parameter.filename, d);
                    double vllh = spn.llh(Partition.Validation);
                    double tllh = spn.llh(Partition.Testing);

                    // print all params separated by tabs
                    System.out.print(data_id+" ");
                    System.out.print(d);
                    System.out.print("\tGF: "+VarInstSplit.gfactor);
                    System.out.print("\tCP: "+VarInstSplit.clusterPenalty);
                    System.out.print("\tValid: "+vllh);
                    System.out.print("\tTest: "+tllh);
                    System.out.println("\tTime: "+1.0*(toc-tic)/1000);
                    
                }else{
                    
                    System.out.println("Learning for just smooth");
                    GraphSPN spn = slalg.learnStructure(d);
                

                    long toc = System.currentTimeMillis();

                    double highestValidation = Double.NEGATIVE_INFINITY;

                    double smoo = 1.0;
                    for(smoo = 2.0; smoo>=0.02; smoo*=0.5){
                            SmoothedMultinomialNode.smooth = smoo; //0.0000001
                            //PolynomaialNode.smooth = smoo;
                            for(Node n : spn.order){
                                    if(n instanceof SmoothedMultinomialNode){
                                            SmoothedMultinomialNode smn = (SmoothedMultinomialNode) n;
                                            smn.reset();
                                    }
                                    /*
                                    for(Node n : spn.order){
                                        if(n instanceof PolynomialNode){
                                            PolynomialNode pn = (PolynomialNode) n;
                                            pn.reset();
                                        }
                                    */
                            }
                            System.out.println("\nCP "+VarInstSplit.clusterPenalty+"\tGF "+VarInstSplit.gfactor+"\tIT "+VarInstSplit.indepInstThresh+"\tSmoo"+SmoothedMultinomialNode.smooth);

                            //				spn.printDepth(10);
                            double llh = spn.llh(Partition.Training);
                            System.out.println("Train LLH: "+llh);

                            llh = spn.llh(Partition.Validation);
                            System.out.println("Validation LLH: "+llh);
                            if(llh > highestValidation){
                                    highestValidation = llh;
                                    spn.write(util.Parameter.filename);
                                    System.out.println("***");
                            }
                    } // end smoothing loop
                    
                    spn = GraphSPN.load(util.Parameter.filename, d);
                    double vllh = spn.llh(Partition.Validation);
                    double tllh = spn.llh(Partition.Testing);

                    // print all params separated by tabs
                    System.out.print(data_id+" ");
                    System.out.print(d);
                    System.out.print("\tGF: "+VarInstSplit.gfactor);
                    System.out.print("\tCP: "+VarInstSplit.clusterPenalty);
                    System.out.print("\tValid: "+vllh);
                    System.out.print("\tTest: "+tllh);
                    System.out.println("\tTime: "+1.0*(toc-tic)/1000);
                
                }

	}


	public static void parseParameters(String[] args){
		int pos = 0;

		while(pos<args.length){

			// Data source
			if(args[pos].equals("DATA")){
				data_id = Integer.parseInt(args[++pos]);
				System.out.println(args[pos-1]+"\t"+args[pos]);
			}
                        
                        if(args[pos].equals("BIN")){
                            bins_perFeature = Integer.parseInt(args[++pos]);
                            System.out.println(args[pos-1]+"\t"+args[pos]);
                        }
			
			// Gfactor
			if(args[pos].equals("GF")){
				VarInstSplit.gfactor = Double.parseDouble(args[++pos]);
				System.out.println(args[pos-1]+"\t"+args[pos]);
			}

			// Cluster penalty
			if(args[pos].equals("CP")){
				VarInstSplit.clusterPenalty = Double.parseDouble(args[++pos]);
				System.out.println(args[pos-1]+"\t"+args[pos]);
			}

			// If there are this many or fewer instances, assume that all variables are independent
			// This just saves us from needlessly computing the pairwise independence test
			if(args[pos].equals("INDEPINST")){
				VarInstSplit.indepInstThresh = Integer.parseInt(args[++pos]);
				System.out.println(args[pos-1]+"\t"+args[pos]);
			}

			// Split MI comp. PLL
			if(args[pos].equals("COMPPLL")){
				VarInstSplit.compPLL = true;
				System.out.println(args[pos]);
			}

			// Filename to save model
			if(args[pos].equals("N")){
				Parameter.filename = args[++pos];
				System.out.println(args[pos-1]+"\t"+args[pos]);
			}

			// Query file
			if(args[pos].equals("Q")){
				queryfile = args[++pos];
				System.out.println(args[pos-1]+"\t"+args[pos]);
			}
			
			// Evidence file
			if(args[pos].equals("EV")){
				evidencefile = args[++pos];
				System.out.println(args[pos-1]+"\t"+args[pos]);
			}
			
                        //Output file
                        if(args[pos].equals("OP")){
                            outputfile = args[++pos];
                            System.out.println(args[pos-1]+"\t"+args[pos]);
                        }
			
			pos++;
		}

	}

}
