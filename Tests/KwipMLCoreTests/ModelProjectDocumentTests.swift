import XCTest
@testable import KwipMLCore

final class ModelProjectDocumentTests:XCTestCase {
    private let projectID="11111111-1111-1111-1111-111111111111"
    private func canonical<T:Encodable>(_ value:T)throws->Data {let e=JSONEncoder();e.outputFormatting=[.prettyPrinted,.sortedKeys];return try e.encode(value)}
    func testLegacyFlatCanonicalBytesAndFaithfulPromotionPreserveIdentity()throws {
        for version in [1,2] {
            var graph=ExampleFlows.parallelClassifier;graph.schemaVersion=version
            if version == 1 {for i in graph.nodes.indices where graph.nodes[i].type == "linear" {graph.nodes[i].config.removeValue(forKey:"bias")}}
            let bytes=try canonical(graph)
            XCTAssertEqual(try ModelProjectCodec.encode(ModelProjectCodec.decode(bytes)),bytes)
            XCTAssertEqual(try ModelProjectCodec.decode(bytes),.legacyFlat(graph))
            let promoted=try ModelProjectCodec.promote(graph,projectID:projectID)
            XCTAssertEqual(promoted.source.schemaVersion,2);XCTAssertEqual(promoted.source.root.configurationVersion,version)
            XCTAssertEqual(promoted.source.root.nodes,graph.nodes);XCTAssertEqual(promoted.source.root.edges,graph.edges)
            XCTAssertEqual(promoted.source.execution,graph.execution);XCTAssertTrue(promoted.drafts.isEmpty)
            XCTAssertEqual(try ModelProjectCodec.decode(ModelProjectCodec.encode(.blockModel(promoted))),.blockModel(promoted))
        }
    }
    func testBoundedIncompleteDraftRoundTripAndExactParentContext()throws {
        let source=CompositeExamples.parallelClassifier,instance=try XCTUnwrap(source.root.instances.first)
        var proposed=try XCTUnwrap(source.definitions.first{$0.reference == instance.definition})
        proposed.body.edges=[];proposed.outputs=[]
        // Missing configuration fields and unfinished connections are editable, not runnable.
        proposed.body.nodes[0].config=[:]
        let draft=ModelBlockDraft(projectID:projectID,instancePath:[instance.id],baseReference:instance.definition,baseRevision:7,proposed:proposed)
        var document=ModelAuthoringDocument(projectID:projectID,revision:7,source:source,drafts:[draft])
        let bytes=try ModelProjectCodec.encode(.blockModel(document))
        XCTAssertEqual(try ModelProjectCodec.decode(bytes),.blockModel(document))
        XCTAssertThrowsError(try CompositeExpander.expand(CompositeProjectDocument(root:proposed.body,definitions:[]),registry:.builtIn()))
        document.drafts[0].baseRevision=6;XCTAssertThrowsError(try ModelProjectCodec.encode(.blockModel(document)))
        document.drafts[0]=draft;document.drafts[0].projectID=UUID().uuidString;XCTAssertThrowsError(try ModelProjectCodec.encode(.blockModel(document)))
        document.drafts[0]=draft;document.drafts[0].instancePath=["missing"];XCTAssertThrowsError(try ModelProjectCodec.encode(.blockModel(document)))
        document.drafts=[draft,draft];XCTAssertThrowsError(try ModelProjectCodec.encode(.blockModel(document)))
    }
    func testStrictTaggedDispatchCannotFallBackAndLimitsAreStorageOnly()throws {
        let document=ModelAuthoringDocument(projectID:projectID,source:CompositeExamples.parallelClassifier)
        let encoded=try ModelProjectCodec.encode(.blockModel(document))
        var root=try JSONSerialization.jsonObject(with:encoded) as! [String:Any]
        root["kind"]="other";XCTAssertThrowsError(try ModelProjectCodec.decode(JSONSerialization.data(withJSONObject:root)))
        root["kind"]="modelProject";root["schemaVersion"]=2;XCTAssertThrowsError(try ModelProjectCodec.decode(JSONSerialization.data(withJSONObject:root)))
        root["schemaVersion"]=1;root["extra"]=true;XCTAssertThrowsError(try ModelProjectCodec.decode(JSONSerialization.data(withJSONObject:root)))
        XCTAssertThrowsError(try ModelProjectCodec.decode(Data(repeating:32,count:2_000_001)))
        var invalid=document;invalid.projectID="not-a-uuid";XCTAssertThrowsError(try ModelProjectCodec.encode(.blockModel(invalid)))
        invalid=document;invalid.source.root.nodes[0].x = .infinity;XCTAssertThrowsError(try ModelProjectCodec.encode(.blockModel(invalid)))
        invalid=document;invalid.source.definitions=Array(repeating:document.source.definitions[0],count:65);XCTAssertThrowsError(try ModelProjectCodec.encode(.blockModel(invalid)))
        invalid=document;invalid.source.root.nodes += Array(repeating:document.source.root.nodes[0],count:129);XCTAssertThrowsError(try ModelProjectCodec.encode(.blockModel(invalid)))
        invalid=document;invalid.source.root.edges=[];XCTAssertNoThrow(try ModelProjectCodec.encode(.blockModel(invalid)))
        invalid=document;invalid.source.root.nodes[0].config=["features":.integer(999999)];XCTAssertNoThrow(try ModelProjectCodec.encode(.blockModel(invalid)))
    }
    func testNestedDraftPathAndMalformedBoundaryMetadata()throws {
        var source=CompositeExamples.parallelClassifier
        let inner=source.root.instances[0].definition
        let outer=CompositeDefinition(reference:.init("outer"),title:"Outer",body:.init(instances:[.init(id:"nested",definition:inner)]),inputs:[.init("in",targets:[.init("nested","in")])],outputs:[.init("out",source:.init("nested","out"))])
        source.definitions.append(outer);source.root.instances[0].definition=outer.reference
        let innerDefinition=try XCTUnwrap(source.definitions.first{$0.reference == inner})
        var draft=ModelBlockDraft(projectID:projectID,instancePath:[source.root.instances[0].id,"nested"],baseReference:inner,baseRevision:0,proposed:innerDefinition)
        XCTAssertNoThrow(try ModelProjectCodec.encode(.blockModel(.init(projectID:projectID,source:source,drafts:[draft]))))
        draft.instancePath=Array(repeating:"nested",count:9)
        XCTAssertThrowsError(try ModelProjectCodec.encode(.blockModel(.init(projectID:projectID,source:source,drafts:[draft]))))
        var invalid=source;invalid.definitions[0].inputs.append(invalid.definitions[0].inputs[0])
        XCTAssertThrowsError(try ModelProjectCodec.encode(.blockModel(.init(projectID:projectID,source:invalid))))
    }
    func testRepeatedDefinitionDAGIsBoundedAndNinthLevelRejected()throws {
        func document(levels:Int)->ModelAuthoringDocument {
            let definitions=(0..<levels).map { level -> CompositeDefinition in
                let instances:[CompositeInstance]=level+1 == levels ? []:(0..<128).map{.init(id:"instance-\($0)",definition:.init("level-\(level+1)"))}
                return CompositeDefinition(reference:.init("level-\(level)"),title:"Level \(level)",body:.init(instances:instances),inputs:[],outputs:[])
            }
            let source=CompositeProjectDocument(root:.init(instances:[.init(id:"root-block",definition:.init("level-0"))]),definitions:definitions)
            return ModelAuthoringDocument(projectID:projectID,source:source)
        }
        // The stored DAG is small; enumerating all instance paths would visit 128^7 leaves.
        // Storage must inspect repeated (definition,depth) contexts once, before execution's flat-count checks.
        let valid=document(levels:8),start=Date()
        let bytes=try ModelProjectCodec.encode(.blockModel(valid))
        XCTAssertLessThan(bytes.count,2_000_000)
        XCTAssertEqual(try ModelProjectCodec.decode(bytes),.blockModel(valid))
        XCTAssertLessThan(Date().timeIntervalSince(start),5,"Bounded source validation must not enumerate expanded instance paths.")
        XCTAssertThrowsError(try ModelProjectCodec.encode(.blockModel(document(levels:9))))
    }

}
