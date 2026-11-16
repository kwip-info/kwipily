import XCTest
@testable import KwipMLCore

final class RetiredTrainingRecipeMigrationTests: XCTestCase {
    private var dataRegistry: DataFlowOperatorRegistry {get throws {
        try .load(from:URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("runner/dataflow-operators.json"))
    }}
    private let address=WorkflowNodeAddress(domain:.control,localID:"retired-optimizer")
    private func legacy() throws -> WorkflowDocument {
        var value=try WorkflowExamples.preparedParallelClassifier(device:"cpu")
        value.schemaVersion=max(13,value.schemaVersion)
        let train=try XCTUnwrap(value.nodes.firstIndex {$0.address == WorkflowExamples.trainAddress})
        value.nodes[train].payload = .train(config:[:])
        value.nodes.append(.init(address:address,name:"Old settings",payload:.optimizerRecipe(type:"optimizer_sgd",configurationVersion:1,config:try OptimizerConfiguration(componentID:"optimizer_sgd",values:["lr":0.031]).values)))
        value.edges.append(.init(id:"old-optimizer-wire",from:.init(address:address,port:"out"),to:.init(address:WorkflowExamples.trainAddress,port:"optimizer"),kind:.optimizerRecipe))
        return value
    }
    private func compile(_ value:WorkflowDocument,historical:Bool=false)throws->WorkflowExecutionPlan {
        if historical {return try WorkflowCompiler.compileHistoricalEvidence(value,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:dataRegistry,contractVersion:13)}
        return try WorkflowCompiler.compile(value,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:dataRegistry,contractVersion:13)
    }
    func testExactUpgradePreservesExecutionWhileCurrentCompilerRefusesOriginal()throws {
        let original=try legacy(),bytes=try WorkflowCodec.encode(original)
        XCTAssertTrue(RetiredTrainingRecipeMigration.requiresUpgrade(original))
        XCTAssertThrowsError(try compile(original))
        XCTAssertThrowsError(try WorkflowCodePlan.compile(original,scope:.training,selectedTargets:[WorkflowExamples.trainAddress],modelRegistry:.builtIn(),dataRegistry:dataRegistry))
        let before=try compile(original,historical:true)
        XCTAssertEqual(before.optimizerRecipe?.source,address)
        XCTAssertFalse(before.inlineTrainingRecipeRoles.contains("optimizer"))
        let upgraded=try RetiredTrainingRecipeMigration.migrate(original),after=try compile(upgraded)
        XCTAssertFalse(RetiredTrainingRecipeMigration.requiresUpgrade(upgraded))
        XCTAssertEqual(before.model,after.model)
        XCTAssertEqual(before.settings,after.settings)
        XCTAssertEqual(before.dataflow,after.dataflow)
        XCTAssertEqual(before.optimizerRecipe?.configuration,after.optimizerRecipe?.configuration)
        XCTAssertEqual(after.optimizerRecipe?.source,WorkflowExamples.trainAddress)
        XCTAssertTrue(after.inlineTrainingRecipeRoles.contains("optimizer"))
        XCTAssertEqual(try WorkflowCodec.encode(original),bytes)
        XCTAssertEqual(try RetiredTrainingRecipeMigration.migrate(upgraded),upgraded)
        let code=try WorkflowCodePlan.compile(upgraded,scope:.training,selectedTargets:[WorkflowExamples.trainAddress],modelRegistry:.builtIn(),dataRegistry:dataRegistry)
        try code.validate()
    }
    func testSharedConsumersCopySettingsAndOrphanConflictRefuseWithoutMutation()throws {
        var source=try legacy()
        let second=WorkflowNodeAddress(domain:.control,localID:"second-train")
        source.nodes.append(.init(address:second,payload:.train(config:[:])))
        source.edges.append(.init(from:.init(address:address,port:"out"),to:.init(address:second,port:"optimizer"),kind:.optimizerRecipe))
        let migrated=try RetiredTrainingRecipeMigration.migrate(source)
        XCTAssertEqual(migrated.nodes.first {$0.address == second}?.payload,migrated.nodes.first {$0.address == WorkflowExamples.trainAddress}?.payload)
        var orphan=source;orphan.edges.removeAll {$0.from.address == address}
        let bytes=try WorkflowCodec.encode(orphan)
        XCTAssertThrowsError(try RetiredTrainingRecipeMigration.migrate(orphan))
        XCTAssertEqual(try WorkflowCodec.encode(orphan),bytes)
        var conflict=try legacy()
        let i=try XCTUnwrap(conflict.nodes.firstIndex {$0.address == WorkflowExamples.trainAddress})
        conflict.nodes[i].payload = .train(config:["optimizer":try OptimizerConfiguration(componentID:"optimizer_sgd",values:[:]).recipeValue])
        XCTAssertThrowsError(try RetiredTrainingRecipeMigration.migrate(conflict))
    }
    func testRepairProjectionPreservesOriginalToolAndSavedWireWithoutNewAuthoringPorts()throws {
        let source=try legacy(),node=try XCTUnwrap(source.nodes.first {$0.address == address})
        let ports=try WorkflowPortContracts.resolveForAuthoring(node,document:source,modelRegistry:.builtIn(),dataRegistry:dataRegistry)
        XCTAssertTrue(ports.inputs.isEmpty);XCTAssertTrue(ports.outputs.isEmpty)
        let train=try XCTUnwrap(source.nodes.first {$0.address == WorkflowExamples.trainAddress})
        XCTAssertEqual(try WorkflowPortContracts.resolveForAuthoring(train,document:source,modelRegistry:.builtIn(),dataRegistry:dataRegistry).inputs.map(\.id),["model","data","configuration"])
        let projection=try WorkflowCanvasProjector.project(source,modelRegistry:.builtIn(),dataRegistry:dataRegistry,inlineTrainControls:true)
        let display=try XCTUnwrap(projection.displayIDByAddress[address])
        let projected=try XCTUnwrap(projection.graph.nodes.first {$0.id == display})
        XCTAssertEqual(projection.registry.component(type:projected.type)?.category,"Repair")
        XCTAssertEqual(projection.registry.component(type:projected.type)?.fields,[])
        XCTAssertEqual(projection.graph.edges.count,source.edges.filter {projection.displayIDByAddress[$0.from.address] != nil && projection.displayIDByAddress[$0.to.address] != nil}.count)
    }
    func testRetiredPlacementAndEditingRefuseButMoveDeleteAllowExplicitRepair()throws {
        let source=try legacy(),node=try XCTUnwrap(source.nodes.first {$0.address == address})
        func apply(_ command:WorkflowCommand,_ doc:WorkflowDocument)throws->WorkflowEditResult {
            try WorkflowEditing.apply(command,to:doc,selection:.init(),modelRegistry:.builtIn(),dataRegistry:dataRegistry)
        }
        XCTAssertThrowsError(try apply(.insert(node),WorkflowDocument(projectID:UUID().uuidString)))
        XCTAssertThrowsError(try apply(.configure(address:address,config:["lr":0.2]),source))
        XCTAssertThrowsError(try apply(.duplicate(addresses:[address],offset:.init(x:1,y:1)),source))
        XCTAssertNoThrow(try apply(.move(addresses:[address],offset:.init(x:2,y:2)),source))
        let repaired=try apply(.delete(.init(nodeAddresses:[address])),source).document
        XCTAssertFalse(RetiredTrainingRecipeMigration.requiresUpgrade(repaired))
    }
}
