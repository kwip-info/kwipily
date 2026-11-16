import XCTest
@testable import KwipMLCore

final class WorkflowEvaluationContractTests:XCTestCase {
    private func document()throws->WorkflowDocument {
        var doc=try WorkflowCodec.migrateToVersion5(WorkflowDocument(projectID:UUID().uuidString))
        doc.nodes=[.init(address:.init(domain:.control,localID:"frozen"),payload:.frozenModel(config:[:])),
                   .init(address:.init(domain:.control,localID:"evaluate"),payload:.evaluateTest(config:[:]))]
        doc.edges=[.init(from:.init(address:doc.nodes[0].address,port:"model"),to:.init(address:doc.nodes[1].address,port:"model"),kind:.frozenModel)]
        return doc
    }
    func testExplicitSchemaFiveRoundTripAndOlderMigrationsNeverDowngrade()throws {
        let doc=try document()
        XCTAssertEqual(doc.schemaVersion,5)
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(doc)),doc)
        XCTAssertEqual(try WorkflowCodec.migrateToVersion2(doc),doc)
        XCTAssertEqual(try WorkflowCodec.migrateToVersion3(doc),doc)
        XCTAssertEqual(try WorkflowCodec.migrateToVersion4(doc),doc)
        XCTAssertEqual(try WorkflowCodec.migrateToVersion5(doc),doc)
        for version in 1...4 {
            var old=doc;old.schemaVersion=version
            XCTAssertThrowsError(try WorkflowCodec.encode(old))
            let bytes=try JSONEncoder().encode(old)
            XCTAssertThrowsError(try WorkflowCodec.decode(bytes))
        }
    }
    func testMigrationPreservesExistingTrainAndRunSetup()throws {
        var source=WorkflowDocument(projectID:UUID().uuidString)
        source.nodes=[.init(address:.init(domain:.control,localID:"setup"),payload:.runSetup(config:["epochs":3,"batchSize":32])),
                      .init(address:.init(domain:.control,localID:"train"),payload:.train(config:[:]))]
        let migrated=try WorkflowCodec.migrateToVersion5(source)
        XCTAssertEqual(migrated.nodes,source.nodes)
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(migrated)),migrated)
    }
    func testFrozenEdgeCannotAppearInAnOlderDocumentWithoutNewPayload()throws {
        var doc=WorkflowDocument(projectID:UUID().uuidString)
        doc.nodes=[.init(address:.init(domain:.control,localID:"a"),payload:.modelDefinition),
                   .init(address:.init(domain:.control,localID:"b"),payload:.modelDefinition)]
        doc.edges=[.init(from:.init(address:doc.nodes[0].address,port:"model"),to:.init(address:doc.nodes[1].address,port:"source"),kind:.frozenModel)]
        for version in 1...4 {doc.schemaVersion=version;XCTAssertThrowsError(try WorkflowCodec.encode(doc))}
    }
    func testPortsKeepFrozenAuthoritySeparateFromEditableModelAndTrain()throws {
        let doc=try document()
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
        let frozen=try WorkflowPortContracts.resolve(doc.nodes[0],document:doc,modelRegistry:.builtIn(),dataRegistry:registry)
        XCTAssertTrue(frozen.inputs.isEmpty)
        XCTAssertEqual(frozen.outputs.map(\.id),["model"]);XCTAssertEqual(frozen.outputs.map(\.kind),[.frozenModel])
        let evaluate=try WorkflowPortContracts.resolve(doc.nodes[1],document:doc,modelRegistry:.builtIn(),dataRegistry:registry)
        XCTAssertEqual(evaluate.inputs.map(\.id),["model","test"])
        XCTAssertEqual(evaluate.inputs.map(\.kind),[.frozenModel,.table])
        XCTAssertTrue(evaluate.inputs.allSatisfy {$0.required && $0.maximumConnections == 1})
        XCTAssertEqual(evaluate.outputs.map(\.id),["predictions"])
        XCTAssertEqual(evaluate.outputs.map(\.kind),[.table])
    }
    func testSettingsDefaultsBoundsAndCodableRoundTrip()throws {
        let defaults=try WorkflowEvaluationSettings.decode([:])
        XCTAssertEqual(defaults,WorkflowEvaluationSettings(device:"auto",batchSize:256,previewLimit:8))
        XCTAssertEqual(try JSONDecoder().decode(WorkflowEvaluationSettings.self,from:JSONEncoder().encode(defaults)),defaults)
        for device in ["auto","cpu","mps"] {
            for batch in [1,100000] {
                for preview in [0,100] {XCTAssertNoThrow(try WorkflowEvaluationSettings(device:device,batchSize:batch,previewLimit:preview).validate())}
            }
        }
        for value in [WorkflowEvaluationSettings(device:"cuda"),.init(batchSize:0),.init(batchSize:100001),.init(previewLimit:-1),.init(previewLimit:101)] {
            XCTAssertThrowsError(try value.validate())
        }
        for config:[String:JSONValue] in [["epochs":1],["seed":42],["learningRate":0.1],["experiment":.object([:])],
            ["batchSize":.number(2.5)],["batchSize":.bool(true)],["device":.integer(1)],["previewLimit":.null]] {
            XCTAssertThrowsError(try WorkflowEvaluationSettings.decode(config))
        }
    }
    func testCodecRejectsFrozenConfigurationAndTrainingControlsOnEvaluation()throws {
        var doc=try document()
        doc.nodes[0].payload = .frozenModel(config:["checkpointPath":.string("/private/model")])
        XCTAssertThrowsError(try WorkflowCodec.encode(doc))
        doc.nodes[0].payload = .frozenModel(config:[:])
        for config:[String:JSONValue] in [["epochs":1],["seed":42],["experiment":.object([:])],["batchSize":0],["previewLimit":101]] {
            doc.nodes[1].payload = .evaluateTest(config:config)
            XCTAssertThrowsError(try WorkflowCodec.encode(doc))
        }
    }
}
