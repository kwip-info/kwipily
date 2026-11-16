import XCTest
import KwipMLCore
@testable import KwipMLApp

final class LossPhaseArgumentsTests:XCTestCase {
    private let root=URL(fileURLWithPath:"/private/kwipml-phase-test"),project=UUID().uuidString,directory=UUID().uuidString
    private func arguments(_ task:LossLearningTask = .mse)throws->[String] {
        let registry=try ComponentRegistry.builtIn(),repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let data=try DataFlowOperatorRegistry.load(from:repo.appendingPathComponent("runner/dataflow-operators.json"))
        let plan=try WorkflowCompiler.compile(WorkflowExamples.preparedLossObjective(task:task),train:WorkflowExamples.trainAddress,modelRegistry:registry,dataRegistry:data,contractVersion:12)
        let recipe=try XCTUnwrap(plan.lossRecipe),run=root.appendingPathComponent("workflow-runs/\(project)/\(directory)")
        // Structural pins here are declared input fixtures, not runtime verification claims.
        let reference=PreparedDataReference(manifestPath:run.appendingPathComponent("preparation/artifacts/manifest.json").path,manifestSHA256:String(repeating:"a",count:64),graphDigest:WorkflowArchiveCodec.hash(try DataFlowCodec.encodeGraph(plan.dataflow)),trainingNode:plan.training.node,trainingPort:plan.training.port,validationNode:plan.validation.node,validationPort:plan.validation.port)
        let selection=PreparedObjectiveSelection(reference:reference,objectiveRecipe:recipe.configuration.recipeValue,targetMapping:recipe.targetMapping,scalarization:.mean)
        return ["--graph",run.appendingPathComponent("training/request.json").path,"--output",run.appendingPathComponent("training/artifacts").path]+(try WorkflowPhaseArtifacts.preparedArguments(.objective(selection)))
    }
    private func validate(_ args:[String])throws {try WorkflowPhaseArguments.validate(args,workspace:root,projectID:project,directoryID:directory,phase:.training)}
    private func replace(_ args:[String],key:String,value:String)->[String] {var result=args;if let index=result.firstIndex(of:key) {result[index+1]=value};return result}
    private func remove(_ args:[String],key:String)->[String] {var result=args;if let index=result.firstIndex(of:key) {result.removeSubrange(index...index+1)};return result}
    func testAllEightActualCanonicalPlanSelectionsAdmitMeanAndSum()throws {
        for task in LossLearningTask.allCases {let args=try arguments(task);XCTAssertNoThrow(try validate(args),task.rawValue);XCTAssertNoThrow(try validate(replace(args,key:"--objective-scalarization",value:"sum")),task.rawValue)}
    }
    func testIncompleteMixedDuplicateUnknownNoncanonicalAndForeignObjectiveFieldsRefuse()throws {
        let args=try arguments()
        for key in ["--objective-recipe-json","--objective-target-mapping-json","--objective-scalarization","--prepared-dataflow","--prepared-manifest-sha256","--prepared-graph-digest","--prepared-training-node","--prepared-training-port","--prepared-validation-node","--prepared-validation-port"] {XCTAssertThrowsError(try validate(remove(args,key:key)),key)}
        XCTAssertThrowsError(try validate(args+["--feature-columns-json","[\"a\"]"]))
        XCTAssertThrowsError(try validate(args+["--target-column","label"]))
        XCTAssertThrowsError(try validate(args+["--objective-scalarization","mean"]))
        XCTAssertThrowsError(try validate(args+["--unknown-objective","x"]))
        XCTAssertThrowsError(try validate(replace(args,key:"--objective-scalarization",value:"average")))
        XCTAssertThrowsError(try validate(replace(args,key:"--prepared-manifest-sha256",value:"unverified")))
        let recipeIndex=try XCTUnwrap(args.firstIndex(of:"--objective-recipe-json"))
        var recipe=try JSONDecoder().decode([String:JSONValue].self,from:Data(args[recipeIndex+1].utf8))
        guard case .object(var config)=recipe["config"] else {return XCTFail("Canonical recipe fixture missing")}
        config.removeValue(forKey:"reduction");recipe["config"] = .object(config)
        XCTAssertThrowsError(try validate(replace(args,key:"--objective-recipe-json",value:String(decoding:WorkflowPhaseArtifacts.encode(recipe),as:UTF8.self))))
        let mappingIndex=try XCTUnwrap(args.firstIndex(of:"--objective-target-mapping-json"))
        var mapping=try JSONDecoder().decode([String:JSONValue].self,from:Data(args[mappingIndex+1].utf8))
        mapping["componentID"] = .string("loss_l1")
        XCTAssertThrowsError(try validate(replace(args,key:"--objective-target-mapping-json",value:String(decoding:WorkflowPhaseArtifacts.encode(mapping),as:UTF8.self))))
        mapping["componentID"] = .string("loss_mse");mapping["unknown"] = .bool(true)
        XCTAssertThrowsError(try validate(replace(args,key:"--objective-target-mapping-json",value:String(decoding:WorkflowPhaseArtifacts.encode(mapping),as:UTF8.self))))
    }
    func testObjectiveOnlyMappingBudgetDoesNotRelaxGenericArgumentOrOwnershipEnvelope()throws {
        var args=try arguments(.bce)
        func names(_ prefix:String,_ unicode:Bool)->[String] {(0..<128).map {prefix+String(format:"%03d",$0)+String(repeating:unicode ? "😀":"x",count:195)}}
        let config=try LossConfiguration(componentID:"loss_bce_with_logits",values:["weight_mode":.string("mapped"),"pos_weight_mode":.string("mapped")])
        func mapping(_ unicode:Bool)->LossTargetMapping {.init(componentID:config.componentID,inputColumns:names("f",unicode),roles:[.init(role:"target",columns:names("t",unicode),sampleShape:[128],dtype:.float32),.init(role:"weight",columns:names("w",unicode),sampleShape:[128],dtype:.float32),.init(role:"pos_weight",columns:names("p",unicode),sampleShape:[128],dtype:.float32)])}
        args=replace(args,key:"--objective-recipe-json",value:String(decoding:try WorkflowPhaseArtifacts.encode(config.recipeValue),as:UTF8.self))
        let ascii=String(decoding:try WorkflowPhaseArtifacts.encode(mapping(false)),as:UTF8.self)
        XCTAssertGreaterThan(ascii.utf8.count,65536);XCTAssertLessThan(ascii.utf8.count,524288)
        args=replace(args,key:"--objective-target-mapping-json",value:ascii);XCTAssertNoThrow(try validate(args))
        XCTAssertThrowsError(try validate(args+["--composite-source",String(repeating:"x",count:65537)]))
        XCTAssertThrowsError(try validate(replace(args,key:"--objective-recipe-json",value:String(repeating:" ",count:32769))))
        XCTAssertThrowsError(try validate(replace(args,key:"--objective-target-mapping-json",value:String(repeating:" ",count:524289))))
        let unicode=String(decoding:try WorkflowPhaseArtifacts.encode(mapping(true)),as:UTF8.self)
        args=replace(args,key:"--objective-target-mapping-json",value:unicode);XCTAssertNoThrow(try validate(args))
        let request=WorkflowOperationRequest(operationID:UUID().uuidString,projectID:project,runID:UUID().uuidString,directoryID:directory,requestIdentity:String(repeating:"b",count:64),phase:.training,arguments:args)
        let bytes=try WorkflowArchiveCodec.encode(request)
        XCTAssertGreaterThan(bytes.count,256000)
        XCTAssertThrowsError(try WorkflowOperationRequest.decode(bytes),"The separate 256000-byte operation envelope remains stronger than phase admission")
    }
    func testPreparationAndLegacyTrainingRetainTheirPreviousLimitsAndDoNotAcceptObjectiveFlags()throws {
        let run=root.appendingPathComponent("workflow-runs/\(project)/\(directory)")
        let legacy=["--graph",run.appendingPathComponent("training/request.json").path,"--output",run.appendingPathComponent("training/artifacts").path,"--feature-columns-json","[\"a\",\"b\"]","--target-column","label"]
        XCTAssertNoThrow(try validate(legacy));XCTAssertThrowsError(try validate(legacy+["--composite-source",String(repeating:"x",count:65537)]))
        let prep=["--execute-dataflow",run.appendingPathComponent("preparation/request.json").path,"--source-bindings",run.appendingPathComponent("preparation/bindings.json").path,"--output",run.appendingPathComponent("preparation/artifacts").path]
        XCTAssertNoThrow(try WorkflowPhaseArguments.validate(prep,workspace:root,projectID:project,directoryID:directory,phase:.preparation))
        XCTAssertThrowsError(try WorkflowPhaseArguments.validate(prep+["--objective-scalarization","mean"],workspace:root,projectID:project,directoryID:directory,phase:.preparation))
    }
}
