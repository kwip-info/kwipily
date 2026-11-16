import XCTest
import KwipMLCore
@testable import KwipMLApp

final class LossWorkflowArchiveTests:XCTestCase {
    private var repository:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    private func compile(_ document:WorkflowDocument,version:Int=12)throws->WorkflowExecutionPlan {
        var registry=try DataFlowOperatorRegistry.load(from:repository.appendingPathComponent("runner/dataflow-operators.json"))
        if version<=7 {
            let retired=try DataFlowOperatorRegistry.load(from:repository.appendingPathComponent("runner/retired-data-operators-v1.json"))
            XCTAssertTrue(Set(registry.operators.map(\.id)).isDisjoint(with:retired.operators.map(\.id)))
            registry.operators += retired.operators
            return try WorkflowCompiler.compileHistoricalEvidence(document,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:registry,contractVersion:version)
        }
        return try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:registry,contractVersion:version)
    }
    private func object<T:Encodable>(_ value:T)throws->[String:Any] {try XCTUnwrap(JSONSerialization.jsonObject(with:WorkflowArchiveCodec.encode(value)) as? [String:Any])}
    private func capture(_ document:WorkflowDocument,version:Int=12)throws->WorkflowArchiveCapture {
        let plan=try compile(document,version:max(2,version)),pin=String(repeating:"a",count:64)
        func file(_ path:String)->WorkflowArchiveFile {.init(path:path,bytes:10,sha256:pin)}
        return CurrentArchiveFixture.current(.init(schemaVersion:document.schemaVersion,compilerContractVersion:version,projectID:document.projectID,directoryID:"22222222-2222-4222-8222-222222222222",runID:"33333333-3333-4333-8333-333333333333",requestIdentity:pin,revision:document.revision,train:plan.train,workflow:file("workflow.json"),modelRegistry:file("model-registry.json"),dataRegistry:file("data-registry.json"),modelRegistrySHA256:pin,dataRegistrySHA256:pin,runtimeEvidence:file("runtime-evidence.json"),sources:plan.sourceAddresses.enumerated().map {.init(address:$0.element,path:"inputs/source-\($0.offset).csv",bytes:10,sha256:pin)},plan:try WorkflowArchivePlan(plan,compilerContractVersion:version)))
    }
    func testEightActualObjectivePlansCaptureQualifiedRecipeMappingAndScalarization()throws {
        XCTAssertEqual(WorkflowArchiveCodec.trainingCompilerContractVersion(forWorkflowSchema:8),7)
        for task in LossLearningTask.allCases {
            let document=try WorkflowExamples.preparedLossObjective(task:task,device:"cpu")
            let plan=try compile(document),archive=try WorkflowArchivePlan(plan,compilerContractVersion:12)
            let expected=try XCTUnwrap(plan.lossRecipe)
            let raw=try object(archive),recipe=try XCTUnwrap(raw["lossRecipe"] as? [String:Any])
            XCTAssertEqual(Set(recipe.keys),["source","configuration","targetMapping"])
            XCTAssertNil(raw["targetColumn"])
            XCTAssertEqual(raw["objectiveScalarization"] as? String,"mean")
            let configuration=try JSONDecoder().decode(JSONValue.self,from:JSONSerialization.data(withJSONObject:try XCTUnwrap(recipe["configuration"])))
            XCTAssertEqual(try LossConfiguration.decodeRecipe(configuration),expected.configuration)
            XCTAssertEqual(try JSONDecoder().decode(WorkflowNodeAddress.self,from:JSONSerialization.data(withJSONObject:try XCTUnwrap(recipe["source"]))),expected.source)
            XCTAssertEqual(try JSONDecoder().decode(LossTargetMapping.self,from:JSONSerialization.data(withJSONObject:try XCTUnwrap(recipe["targetMapping"]))),expected.targetMapping)
            let captured=try capture(document),bytes=try WorkflowArchiveCodec.encodeCapture(captured)
            let decoded=try WorkflowArchiveCodec.decodeCapture(bytes)
            XCTAssertEqual(try WorkflowArchiveCodec.encodeCapture(decoded),bytes)
            XCTAssertTrue(try decoded.plan.matches(plan,compilerContractVersion:12))
        }
    }
    func testRecompiledObjectiveChangesCannotMatchCapturedPlan()throws {
        let document=try WorkflowExamples.preparedL1Regression(device:"cpu")
        let plan=try compile(document),archive=try WorkflowArchivePlan(plan,compilerContractVersion:12)
        let recipe=try XCTUnwrap(plan.lossRecipe)
        var changed=plan
        changed.lossRecipe=try .init(configuration:.init(componentID:recipe.configuration.componentID,values:["reduction":.string("sum")]),source:recipe.source,targetMapping:recipe.targetMapping)
        XCTAssertFalse(try archive.matches(changed,compilerContractVersion:12))
        changed=plan;changed.lossRecipe=try .init(configuration:recipe.configuration,source:.init(domain:.control,localID:"different-objective"),targetMapping:recipe.targetMapping)
        XCTAssertThrowsError(try archive.matches(changed,compilerContractVersion:12),"A forged inline origin is refused before comparison")
        var mapping=recipe.targetMapping;mapping.roles[0].columns.reverse()
        changed=plan;changed.lossRecipe=try .init(configuration:recipe.configuration,source:recipe.source,targetMapping:mapping)
        XCTAssertFalse(try archive.matches(changed,compilerContractVersion:12))
        changed=plan;changed.objectiveScalarization = .sum
        XCTAssertFalse(try archive.matches(changed,compilerContractVersion:12))
        for version in 1...6 {XCTAssertThrowsError(try WorkflowArchivePlan(plan,compilerContractVersion:version))}
    }
    func testMalformedObjectiveCaptureRefusesVersionsSourcesControlsMappingAndScalarization()throws {
        let captured=try capture(WorkflowExamples.preparedMSERegression(device:"cpu"))
        for mutation in ["unknownWrapper","unknownEnvelope","unknownControl","missingControl","foreignComponent","recipeVersion","configurationVersion","wrongDomain","blankSource","longSource","trainSource","optimizerSourceAlias","mappingVersion","mappingComponent","unknownMapping","missingRole","wrongDType","wrongFeatureColumns","targetInjection","missingScalar","invalidScalar","oldPair","oversize"] {
            var raw=try object(captured),plan=try XCTUnwrap(raw["plan"] as? [String:Any]),recipe=try XCTUnwrap(plan["lossRecipe"] as? [String:Any])
            var config=try XCTUnwrap(recipe["configuration"] as? [String:Any]),mapping=try XCTUnwrap(recipe["targetMapping"] as? [String:Any]),source=try XCTUnwrap(recipe["source"] as? [String:Any])
            switch mutation {
            case "unknownWrapper":recipe["unknownPolicy"]=true
            case "unknownEnvelope":config["unknownPolicy"]=true
            case "unknownControl":var controls=try XCTUnwrap(config["config"] as? [String:Any]);controls["unknownPolicy"]=true;config["config"]=controls
            case "missingControl":var controls=try XCTUnwrap(config["config"] as? [String:Any]);controls.removeValue(forKey:"target_broadcast");config["config"]=controls
            case "foreignComponent":config["componentID"]="optimizer_adam"
            case "recipeVersion":config["schemaVersion"]=2
            case "configurationVersion":config["configurationVersion"]=2
            case "wrongDomain":source["domain"]="model"
            case "blankSource":source["localID"]=" "
            case "longSource":source["localID"]=String(repeating:"a",count:129)
            case "trainSource":source=try XCTUnwrap(plan["train"] as? [String:Any]);plan.removeValue(forKey:"inlineTrainingRecipeRoles")
            case "optimizerSourceAlias":
                let selected=try OptimizerConfiguration(componentID:"optimizer_sgd",values:["lr":.number(captured.plan.settings.learningRate)])
                plan["optimizerRecipe"]=["source":source,"configuration":try object(selected.recipeValue)]
            case "mappingVersion":mapping["schemaVersion"]=2
            case "mappingComponent":mapping["componentID"]="loss_l1"
            case "unknownMapping":mapping["unknownPolicy"]=true
            case "missingRole":mapping["roles"]=[]
            case "wrongDType":var roles=try XCTUnwrap(mapping["roles"] as? [[String:Any]]);roles[0]["dtype"]="int64";mapping["roles"]=roles
            case "wrongFeatureColumns":plan["featureColumns"]=["foreign"]
            case "targetInjection":plan["targetColumn"]="dummy"
            case "missingScalar":plan.removeValue(forKey:"objectiveScalarization")
            case "invalidScalar":plan["objectiveScalarization"]="hiddenMean"
            case "oldPair":raw["schemaVersion"]=7;raw["compilerContractVersion"]=6
            default:var controls=try XCTUnwrap(config["config"] as? [String:Any]);controls["overflow"]=String(repeating:"x",count:32769);config["config"]=controls
            }
            recipe["source"]=source;recipe["configuration"]=config;recipe["targetMapping"]=mapping;plan["lossRecipe"]=recipe;raw["plan"]=plan
            XCTAssertThrowsError(try WorkflowArchiveCodec.decodeCapture(JSONSerialization.data(withJSONObject:raw)),mutation)
        }
    }
    func testCanonicalObjectiveRecipeByteBudgetCannotBeBypassedByCaptureEnvelope()throws {
        let captured=try capture(WorkflowExamples.preparedBCEMultilabel(device:"cpu"))
        var raw=try object(captured),plan=try XCTUnwrap(raw["plan"] as? [String:Any]),recipe=try XCTUnwrap(plan["lossRecipe"] as? [String:Any])
        var configuration=try XCTUnwrap(recipe["configuration"] as? [String:Any]),controls=try XCTUnwrap(configuration["config"] as? [String:Any])
        controls["weight_mode"]="configured";controls["weight_shape"]=[4096];controls["weight_values"]=Array(repeating:0.1234567890123456,count:4096)
        configuration["config"]=controls;recipe["configuration"]=configuration;plan["lossRecipe"]=recipe;raw["plan"]=plan
        XCTAssertThrowsError(try WorkflowArchiveCodec.decodeCapture(JSONSerialization.data(withJSONObject:raw))) {error in
            XCTAssertTrue(error.localizedDescription.contains("32768"),error.localizedDescription)
        }
    }
    func testCurrentInlineObjectiveCoexistsWithOptimizerAndSchedulerInCurrentCapture()throws {
        var document=try WorkflowExamples.preparedStepLRClassifier(device:"cpu")
        let prior=try compile(document),target=try XCTUnwrap(prior.targetColumn)
        let mapping=LossTargetMapping(componentID:"loss_cross_entropy",inputColumns:prior.featureColumns,roles:[.init(role:"target",columns:[target],sampleShape:[],dtype:.int64)])
        let data=try XCTUnwrap(document.nodes.firstIndex {if case .trainingData = $0.payload {return true};return false})
        document.nodes[data].payload = .trainingData(config:["lossTargetMapping":try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(mapping))])
        let train=try XCTUnwrap(document.nodes.firstIndex {$0.address==prior.train})
        guard case .train(var config)=document.nodes[train].payload else {throw WorkflowArchiveError("Missing Train")}
        config["objective"]=try LossConfiguration(componentID:"loss_cross_entropy",values:[:]).recipeValue
        document.nodes[train].payload = .train(config:config)
        let plan=try compile(document),captured=try capture(document)
        XCTAssertNotNil(plan.optimizerRecipe);XCTAssertNotNil(plan.schedulerRecipe);XCTAssertNotNil(plan.lossRecipe)
        XCTAssertEqual(plan.optimizerRecipe?.source,plan.train);XCTAssertEqual(plan.schedulerRecipe?.source,plan.train);XCTAssertEqual(plan.lossRecipe?.source,plan.train)
        XCTAssertEqual(plan.inlineTrainingRecipeRoles,["optimizer","scheduler","objective"])
        let bytes=try WorkflowArchiveCodec.encodeCapture(captured),decoded=try WorkflowArchiveCodec.decodeCapture(bytes)
        XCTAssertEqual(try WorkflowArchiveCodec.encodeCapture(decoded),bytes)
        XCTAssertTrue(try decoded.plan.matches(compile(document),compilerContractVersion:12))
        XCTAssertNil(try object(decoded.plan)["targetColumn"])
    }
    func testHistoricalCanonicalCapturesAndImplicitCompilerSevenOmitObjectiveFields()throws {
        for schema in 1...8 {
            var document=try HistoricalLossArchiveSource.preparedParallelClassifier(projectID:"11111111-1111-4111-8111-111111111111",device:"cpu");document.schemaVersion=schema
            let version = schema == 8 ? 7:WorkflowArchiveCodec.trainingCompilerContractVersion(forWorkflowSchema:schema)
            let value=try capture(document,version:version),bytes=try WorkflowArchiveCodec.encodeCapture(value)
            let raw=try object(value),plan=try XCTUnwrap(raw["plan"] as? [String:Any])
            XCTAssertNil(plan["lossRecipe"]);XCTAssertNil(plan["objectiveScalarization"])
            XCTAssertNotNil(plan["targetColumn"])
            XCTAssertEqual(try WorkflowArchiveCodec.encodeCapture(WorkflowArchiveCodec.decodeCapture(bytes)),bytes)
            // Captured from the actual pre-objective archive implementation during semantic RED.
            let historicalHashes: [Int:String] = [1: "51ce36147b3c001a963eb79d29199e5e2ea7a5004fcef64d1353b64337991f04", 2: "13fe88c722e795ca34acb679bc238052cac46afac09dc897d257689b7c9f347f", 3: "226dda87d01359596033a0d6e6c2140bcac8177d7d9013e59e1f0bbadf3963a0", 4: "ed0da9510d085d1e0b7630a9504c4c760e2519e0d3e92ada78046cf1b959d82f", 5: "45fb520d427e3f8cbd73310989adc6fb5debc57b34ea062353f5a94892e5baab", 6: "07802dce9c549621f50f1913efb9c5154e105ad544c1ef33c5ebb2d99fd5ef20", 7: "fe064145865d3591080149144776854fabb694b1420464d4ae7cd3043ed19b4c"]
            if let expected=historicalHashes[schema] {XCTAssertEqual(WorkflowArchiveCodec.hash(bytes),expected)}
            var malformed=raw,changed=plan;changed["objectiveScalarization"]="mean";malformed["plan"]=changed
            XCTAssertThrowsError(try WorkflowArchiveCodec.decodeCapture(JSONSerialization.data(withJSONObject:malformed)))
        }
    }
}
