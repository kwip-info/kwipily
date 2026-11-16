import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class GradientRunTransportTests:XCTestCase {
    private var discoveryWorkspace:WorkspaceStore?
    private var repository:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    private func setup(_ task:GradientLearningTask, selected:Bool=true, freezeUpdates:Bool=false)async throws->(URL,WorkflowAuthoringStore,RuntimeLocation) {
        let root=URL(fileURLWithPath:FileManager.default.temporaryDirectory.resolvingSymlinksInPath().path).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        var doc=try GradientWorkflowExamples.document(task:task,device:"cpu")
        if freezeUpdates {
            let layer=try XCTUnwrap(doc.nodes.firstIndex {$0.address == .init(domain:.model,localID:"first")})
            var parameters=doc.nodes[layer].parameterSettings ?? [:]
            parameters["weight"] = .object(["trainability":.string("frozen")])
            doc.nodes[layer].parameterSettings=parameters
        }
        if !selected {
            let train=try XCTUnwrap(doc.nodes.firstIndex {$0.address == WorkflowExamples.trainAddress})
            guard case .train(var settings)=doc.nodes[train].payload else {throw WorkflowPhaseError("Missing Train")}
            settings.removeValue(forKey:"gradientPolicy");doc.nodes[train].payload = .train(config:settings)
        }
        let library=ProjectLibrary(directory:root.appendingPathComponent("projects"))
        _ = try library.saveSource(.workflow(doc),id:doc.projectID)
        let owner=try WorkflowAuthoringStore(library:library,projectID:doc.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repository.appendingPathComponent("runner/dataflow-operators.json")))
        let runtime=try RuntimeLocation.resolve(resources:nil,developmentRoot:repository,allowDevelopment:true)
        if discoveryWorkspace == nil {
            let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root.appendingPathComponent("capability-session"),detectCapabilities:false)
            await verifyRuntimeForIntegration(workspace)
            XCTAssertEqual(workspace.runtime,runtime)
            _ = try XCTUnwrap(workspace.moduleCompositionDiscoveryEvidence,"Fresh capture requires actual selected-worker discovery")
            discoveryWorkspace=workspace
        }
        return(root,owner,runtime)
    }
    private var budget:ResolvedExecutionBudget {get throws {try .resolve(global:.factory,projectMaximumRunSeconds:60)}}
    private func capture(_ owner:WorkflowAuthoringStore,_ runtime:RuntimeLocation,_ permission:DerivativeCapturePolicy?,budget:ResolvedExecutionBudget?=nil)throws->WorkflowExecutionRequest {
        try .capture(from:owner,train:WorkflowExamples.trainAddress,runtime:runtime,executionBudget:budget,derivativeCapture:permission,moduleCompositionEvidence:try XCTUnwrap(discoveryWorkspace?.moduleCompositionDiscoveryEvidence))
    }
    func testRequestedValuesRequireIndependentCapturedPermissionAndBudgetBeforeSourceStaging()async throws {
        let(root,owner,runtime)=try await setup(.selectedInfluence);defer{try? FileManager.default.removeItem(at:root)}
        XCTAssertThrowsError(try capture(owner,runtime,nil,budget:budget))
        XCTAssertThrowsError(try capture(owner,runtime,.factory,budget:budget))
        XCTAssertThrowsError(try capture(owner,runtime,.init(values:true)))
        let request=try capture(owner,runtime,.init(values:true),budget:budget)
        XCTAssertEqual(request.derivativeCapture,.init(values:true))
        XCTAssertEqual(request.plan.gradientPolicyRecipe?.configuration.requests.count,1)
        XCTAssertEqual(owner.document,request.document)
    }
    func testEmptyRequestPolicyCapturesFalseAndTrueIndependentlyButAbsencePreservesIdentity()async throws {
        let(root,owner,runtime)=try await setup(.combineSmallGroups);defer{try? FileManager.default.removeItem(at:root)}
        let off=try capture(owner,runtime,.factory,budget:budget),on=try capture(owner,runtime,.init(values:true),budget:budget)
        XCTAssertEqual(off.derivativeCapture,.factory);XCTAssertEqual(on.derivativeCapture,.init(values:true))
        XCTAssertNotEqual(off.requestIdentity,on.requestIdentity)
        XCTAssertEqual(off.document,on.document);XCTAssertEqual(off.plan,on.plan)
        XCTAssertEqual(try capture(owner,runtime,.factory,budget:budget).requestIdentity,off.requestIdentity)
        let(oldRoot,oldOwner,oldRuntime)=try await setup(.combineSmallGroups,selected:false);defer{try? FileManager.default.removeItem(at:oldRoot)}
        let implicit=try capture(oldOwner,oldRuntime,nil,budget:budget),unused=try capture(oldOwner,oldRuntime,.init(values:true),budget:budget)
        XCTAssertNil(unused.derivativeCapture);XCTAssertEqual(implicit.requestIdentity,unused.requestIdentity)
    }
    func testExperimentVariantsRetainCapturedPermissionAndParentIdentity()async throws {
        let(root,owner,runtime)=try await setup(.combineSmallGroups);defer{try? FileManager.default.removeItem(at:root)}
        let spec=WorkflowExperimentSpecification(axes:[.init(field:"seed",values:[41,42])],budget:.init(maximumRuns:2,maximumTotalEpochs:4))
        XCTAssertTrue(owner.setExperimentSpecification(spec,for:.init(domain:.control,localID:"setup"),lease:owner.lease()))
        let base=try capture(owner,runtime,.init(values:true),budget:budget)
        let owned=try WorkflowGroupOwnership.createGroup(workspace:root,projectID:owner.projectID);defer{owned.owner.close()}
        let inputs=try WorkflowExperimentInputs.captureBytes(addresses:base.plan.sourceAddresses,directory:owned.directory) {_ in Data(try GradientWorkflowExamples.sourceCSV().utf8)}
        let variants=try base.experimentVariants(inputs:inputs,owner:owner)
        XCTAssertEqual(base.derivativeCapture,.init(values:true));XCTAssertEqual(variants.count,2);XCTAssertTrue(variants.allSatisfy {$0.derivativeCapture == base.derivativeCapture})
        XCTAssertTrue(variants.allSatisfy {$0.experiment?.sourceRequestIdentity == base.requestIdentity})
        XCTAssertTrue(variants.allSatisfy {$0.plan.gradientPolicyRecipe == base.plan.gradientPolicyRecipe})
    }
    func testLaunchArgumentsUseOnlyCapturedPoliciesAndIndependentPermission()async throws {
        let(root,owner,runtime)=try await setup(.combineSmallGroups,freezeUpdates:true);defer{try? FileManager.default.removeItem(at:root)}
        let request=try capture(owner,runtime,.factory,budget:budget)
        let args=try WorkflowGradientRunArguments.arguments(for:request)
        let trainability=try XCTUnwrap(request.plan.trainabilityRecipe)
        XCTAssertEqual(trainability.configuration.owners,[.object(["node":.string("first"),"parameter":.string("weight"),"mode":.string("frozen")])])
        func value(_ flag:String)throws->String {args[try XCTUnwrap(args.firstIndex(of:flag))+1]}
        XCTAssertEqual(try JSONDecoder().decode(JSONValue.self,from:Data(value("--gradient-policy-json").utf8)),request.plan.gradientPolicyRecipe?.configuration.recipeValue)
        XCTAssertEqual(try JSONDecoder().decode(JSONValue.self,from:Data(value("--trainability-json").utf8)),request.plan.trainabilityRecipe?.configuration.recipeValue)
        XCTAssertEqual(try DerivativeCapturePolicy.decode(Data(value("--derivative-capture-json").utf8)),.factory)
        var invalid=request;invalid.derivativeCapture=nil
        XCTAssertThrowsError(try WorkflowGradientRunArguments.arguments(for:invalid))
        let(oldRoot,oldOwner,oldRuntime)=try await setup(.combineSmallGroups,selected:false,freezeUpdates:true);defer{try? FileManager.default.removeItem(at:oldRoot)}
        let old=try capture(oldOwner,oldRuntime,.init(values:true),budget:budget)
        let oldArgs=try WorkflowGradientRunArguments.arguments(for:old)
        XCTAssertEqual(oldArgs.first,"--trainability-json");XCTAssertEqual(oldArgs.count,2)
    }
    func testTrainingPhaseAcceptsExactBoundedPolicyPermissionAndTrainabilityOnly()throws {
        let root=URL(fileURLWithPath:"/private/kwip-gradient-phase"),project=UUID().uuidString,directory=UUID().uuidString
        let path=root.appendingPathComponent("workflow-runs/\(project)/\(directory)")
        let base=["--graph",path.appendingPathComponent("training/request.json").path,"--output",path.appendingPathComponent("training/artifacts").path]
        func wire(_ v:JSONValue)throws->String {String(decoding:try WorkflowPhaseArtifacts.encode(v),as:UTF8.self)}
        let policy=try wire(GradientPolicyConfiguration(values:GradientLearningTask.selectedInfluence.gradientValues).recipeValue)
        let flags=try wire(TrainabilityConfiguration(values:GradientLearningTask.keepSettings.trainabilityValues).recipeValue)
        let yes=String(decoding:try DerivativeCapturePolicy(values:true).wireData(),as:UTF8.self)
        let no=String(decoding:try DerivativeCapturePolicy.factory.wireData(),as:UTF8.self)
        func validate(_ extra:[String])throws {try WorkflowPhaseArguments.validate(base+extra,workspace:root,projectID:project,directoryID:directory,phase:.training)}
        let all=["--gradient-policy-json",policy,"--derivative-capture-json",yes,"--trainability-json",flags]
        XCTAssertNoThrow(try validate(all));XCTAssertNoThrow(try validate(["--trainability-json",flags]))
        for extra in [["--gradient-policy-json",policy],["--gradient-policy-json",policy,"--derivative-capture-json",no],["--derivative-capture-json",yes],all+["--trainability-json",flags],["--gradient-policy-json",String(repeating:" ",count:32769),"--derivative-capture-json",yes],["--trainability-json",String(repeating:" ",count:32769)],["--gradient-policy-json",policy,"--derivative-capture-json","{\"schemaVersion\":1.0,\"values\":true}"],["--gradient-policy-json",policy,"--derivative-capture-json","{\"schemaVersion\":1,\"values\":false,\"values\":true}"]] {XCTAssertThrowsError(try validate(extra))}
        let prep=["--execute-dataflow",path.appendingPathComponent("preparation/request.json").path,"--source-bindings",path.appendingPathComponent("preparation/bindings.json").path,"--output",path.appendingPathComponent("preparation/artifacts").path]
        XCTAssertThrowsError(try WorkflowPhaseArguments.validate(prep+all,workspace:root,projectID:project,directoryID:directory,phase:.preparation))
    }
}
