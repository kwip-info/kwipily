import XCTest
import CryptoKit
@testable import KwipMLCore

final class ModelExecutionSnapshotTests:XCTestCase {
    private func hash(_ graph:GraphDocument,_ registry:ComponentRegistry)throws->String {
        SHA256.hash(data:try GraphSemantics.canonicalData(graph,registry:registry)).map{String(format:"%02x",$0)}.joined()
    }
    func testLegacyFlatSnapshotPreservesRawIdentityAndExistingSemanticDigest()throws {
        let registry=try ComponentRegistry.builtIn()
        var graph=ExampleFlows.parallelClassifier;graph.schemaVersion=1
        for i in graph.nodes.indices where graph.nodes[i].type=="linear" {graph.nodes[i].config.removeValue(forKey:"bias")}
        graph.nodes[1].name="User label";graph.execution=ExecutionSettings(seed:13)
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
        let raw=try encoder.encode(graph)
        let snapshot=try ModelExecutionSnapshotBuilder.build(.legacyFlat(graph),registry:registry)
        XCTAssertEqual(snapshot.graph,graph);XCTAssertEqual(try encoder.encode(snapshot.graph),raw)
        XCTAssertNil(snapshot.provenance);XCTAssertNil(snapshot.authoringRevision)
        XCTAssertEqual(snapshot.semanticSHA256,try hash(graph,registry))
        XCTAssertEqual(Set(snapshot.origins.keys),Set(graph.nodes.map(\.id)))
        for node in graph.nodes {XCTAssertEqual(snapshot.origins[node.id],CompositeNodeOrigin(instancePath:[],localNodeID:node.id,definition:nil))}
        graph.nodes[1].name="Changed after snapshot"
        XCTAssertEqual(snapshot.graph.nodes[1].name,"User label")
    }
    func testPinnedCanonicalSHAForKnownFlatFixture()throws {
        let graph=GraphDocument(nodes:[GraphNode(id:"input",type:"input",config:["features":2]),GraphNode(id:"linear",type:"linear",config:["in_features":2,"out_features":2]),GraphNode(id:"output",type:"output")],edges:[GraphEdge(id:"one",source:"input",target:"linear"),GraphEdge(id:"two",source:"linear",target:"output")])
        let result=try ModelExecutionSnapshotBuilder.build(.legacyFlat(graph),registry:.builtIn())
        XCTAssertEqual(result.semanticSHA256,"b9dba3b5956c0bd4c3b691fe40955df46a7c736cc4d63dd144e9e73928fa4f21")
    }
    func testNestedBlockSnapshotUsesExistingExpansionIDsAndExactProvenance()throws {
        let registry=try ComponentRegistry.builtIn()
        var source=CompositeExamples.parallelClassifier
        let inner=source.definitions[0].reference
        let wrapper=CompositeDefinition(reference:CompositeDefinitionReference("wrapper"),title:"Wrapper",body:CompositeGraph(instances:[CompositeInstance(id:"inner",definition:inner)]),inputs:[CompositeInput("in",targets:[CompositePortReference("inner","in")])],outputs:[CompositeOutput("out",source:CompositePortReference("inner","out"))])
        source.definitions.append(wrapper);source.root.instances[0].definition=wrapper.reference
        let authored=ModelAuthoringDocument(projectID:UUID().uuidString,revision:17,source:source)
        let before=try CompositeExpander.expand(source,registry:registry)
        let snapshot=try ModelExecutionSnapshotBuilder.build(.blockModel(authored),registry:registry)
        XCTAssertEqual(snapshot.graph,before.graph);XCTAssertEqual(snapshot.origins,before.nodeOrigins)
        XCTAssertEqual(snapshot.authoringRevision,17)
        XCTAssertEqual(snapshot.provenance,CompositeSourceSnapshot(project:source,expansion:before))
        XCTAssertEqual(snapshot.semanticSHA256,try hash(before.graph,registry))
        XCTAssertEqual(snapshot.origins.values.filter{$0.instancePath==["features","inner"]}.count,5)
        XCTAssertEqual(try ModelExecutionSnapshotBuilder.build(.blockModel(authored),registry:registry),snapshot)
    }
    func testUnresolvedDraftAndInvalidExecutionCannotProduceSnapshot()throws {
        let registry=try ComponentRegistry.builtIn(),id=UUID().uuidString
        let source=CompositeExamples.parallelClassifier
        let definition=source.definitions[0]
        let draft=ModelBlockDraft(projectID:id,instancePath:["features"],baseReference:definition.reference,baseRevision:3,proposed:definition)
        let authoring=ModelAuthoringDocument(projectID:id,revision:3,source:source,drafts:[draft])
        XCTAssertThrowsError(try ModelExecutionSnapshotBuilder.build(.blockModel(authoring),registry:registry)) {error in XCTAssertTrue(error.localizedDescription.lowercased().contains("draft"))}
        XCTAssertThrowsError(try ModelExecutionSnapshotBuilder.build(.legacyFlat(GraphDocument()),registry:registry))
        var broken=source;broken.root.edges.removeLast()
        XCTAssertThrowsError(try ModelExecutionSnapshotBuilder.build(.blockModel(ModelAuthoringDocument(projectID:id,source:broken)),registry:registry))
    }
    func testAuthoringSnapshotAcceptedByActualWorkerAndExport()throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let python=root.appendingPathComponent(".venv/bin/python")
        guard FileManager.default.fileExists(atPath:python.path) else {throw XCTSkip("Local Python runtime unavailable")}
        let temp=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:temp,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:temp)}
        let authored=ModelAuthoringDocument(projectID:UUID().uuidString,revision:9,source:CompositeExamples.parallelClassifier)
        let snapshot=try ModelExecutionSnapshotBuilder.build(.blockModel(authored),registry:.builtIn())
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys,.withoutEscapingSlashes]
        let graphURL=temp.appendingPathComponent("request.json"),sourceURL=temp.appendingPathComponent("source.json"),output=temp.appendingPathComponent("artifacts")
        try encoder.encode(snapshot.graph).write(to:graphURL)
        let sourceBytes=try encoder.encode(XCTUnwrap(snapshot.provenance))
        try sourceBytes.write(to:sourceURL)
        let sourceHash=SHA256.hash(data:sourceBytes).map{String(format:"%02x",$0)}.joined()
        func execute(_ arguments:[String])throws {
            let process=Process(),pipe=Pipe();process.executableURL=python;process.currentDirectoryURL=root
            process.arguments=arguments;process.standardOutput=pipe;process.standardError=pipe
            try process.run();let bytes=pipe.fileHandleForReading.readDataToEndOfFile();process.waitUntilExit()
            XCTAssertEqual(process.terminationStatus,0,String(decoding:bytes,as:UTF8.self))
        }
        try execute([root.appendingPathComponent("runner/main.py").path,"--graph",graphURL.path,"--output",output.path,"--device","cpu","--epochs","3","--composite-source",sourceURL.path,"--composite-source-sha256",sourceHash])
        XCTAssertEqual(try Data(contentsOf:output.appendingPathComponent("composite/source.json")),sourceBytes)
        let manifest=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:output.appendingPathComponent("manifest.json"))) as? [String:Any])
        XCTAssertEqual(manifest["status"] as? String,"completed")
        let provenance=try XCTUnwrap(manifest["provenance"] as? [String:Any]),composite=try XCTUnwrap(provenance["composite"] as? [String:Any])
        XCTAssertEqual(composite["sha256"] as? String,sourceHash)
        XCTAssertEqual(composite["expandedSemanticDigest"] as? String,snapshot.semanticSHA256)
        let script="""
        import importlib.util,json,sys,torch
        from pathlib import Path
        from runner.engine import GraphModel,load_data
        root=Path(sys.argv[1]);graph=json.loads((root/'graph.json').read_text())
        checkpoint=torch.load(root/'checkpoint.pt',map_location='cpu',weights_only=True)
        worker=GraphModel(graph);worker.load_state_dict(checkpoint['modelState'],strict=True)
        spec=importlib.util.spec_from_file_location('exported',root/'model.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        exported=module.Model();exported.load_state_dict(checkpoint['modelState'],strict=True)
        assert set(exported.state_dict())==set(worker.state_dict())
        inputs=load_data()[0][2][:8]
        with torch.no_grad():torch.testing.assert_close(exported(inputs),worker(inputs),rtol=0,atol=0)
        assert len(json.loads((root/'metrics.json').read_text()))==3
        """
        try execute(["-c",script,output.path])
    }

}
