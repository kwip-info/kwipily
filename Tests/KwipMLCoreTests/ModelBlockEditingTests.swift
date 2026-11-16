import XCTest
@testable import KwipMLCore

final class ModelBlockEditingTests:XCTestCase {
    private let id="11111111-1111-1111-1111-111111111111"
    private func model(nested:Bool=false)->ModelAuthoringDocument {
        var source=CompositeExamples.parallelClassifier
        if nested {
            let ref=source.root.instances[0].definition
            let wrapper=CompositeDefinition(reference:.init("wrapper"),title:"Wrapper",body:.init(instances:[.init(id:"nested",definition:ref)]),inputs:[.init("in",targets:[.init("nested","in")])],outputs:[.init("out",source:.init("nested","out"))])
            source.definitions.append(wrapper);source.root.instances[0].definition=wrapper.reference
        }
        return ModelAuthoringDocument(projectID:id,revision:4,source:source)
    }
    func testDraftLifecycleRejectsStaleContextAndPreservesIncompleteEdits()throws {
        let original=model(),path=[original.source.root.instances[0].id]
        var begun=try ModelBlockEditing.beginDraft(original,path:path)
        XCTAssertEqual(begun.revision,4);XCTAssertEqual(begun.drafts.count,1)
        XCTAssertEqual(try ModelBlockEditing.beginDraft(begun,path:path),begun)
        var proposal=begun.drafts[0].proposed;proposal.body.edges=[]
        begun=try ModelBlockEditing.updateDraft(begun,proposed:proposal,expectedBaseRevision:4,path:path)
        XCTAssertEqual(begun.drafts[0].proposed.body.edges,[])
        XCTAssertThrowsError(try ModelBlockEditing.applyDraft(begun,path:path,expectedBaseRevision:4,idProvider:{"copy"}))
        XCTAssertThrowsError(try ModelBlockEditing.updateDraft(begun,proposed:proposal,expectedBaseRevision:3,path:path))
        XCTAssertThrowsError(try ModelBlockEditing.discardDraft(begun,path:["missing"],expectedBaseRevision:4))
        let discarded=try ModelBlockEditing.discardDraft(begun,path:path,expectedBaseRevision:4)
        XCTAssertEqual(discarded,original)
    }
    func testSelectedInstanceCopyOnWritePreservesOtherInstanceAndPrimitiveIDs()throws {
        var original=model();let selected=original.source.root.instances[0]
        original.source.root.instances.append(.init(id:"other",definition:selected.definition))
        let before=try CompositeExpander.expand(model().source,registry:.builtIn())
        var begun=try ModelBlockEditing.beginDraft(original,path:[selected.id])
        var proposed=begun.drafts[0].proposed;proposed.title="Local edit"
        let index=try XCTUnwrap(proposed.body.nodes.firstIndex{$0.type == "relu"});proposed.body.nodes[index].type="tanh"
        begun=try ModelBlockEditing.updateDraft(begun,proposed:proposed,expectedBaseRevision:4,path:[selected.id])
        let result=try ModelBlockEditing.applyDraft(begun,path:[selected.id],expectedBaseRevision:4,idProvider:{"private-copy"})
        XCTAssertEqual(result.revision,5);XCTAssertTrue(result.drafts.isEmpty)
        XCTAssertEqual(result.source.root.instances[0].id,selected.id)
        XCTAssertEqual(result.source.root.instances[0].overrides,selected.overrides)
        XCTAssertEqual(result.source.root.instances[1],original.source.root.instances[1])
        XCTAssertEqual(Array(result.source.definitions.prefix(original.source.definitions.count)),original.source.definitions)
        XCTAssertEqual(result.source.root.instances[0].definition.id,"private-copy")
        var runnable=result.source;runnable.root.instances.removeLast()
        let after=try CompositeExpander.expand(runnable,registry:.builtIn())
        XCTAssertEqual(Set(before.graph.nodes.map(\.id)),Set(after.graph.nodes.map(\.id)))
        XCTAssertEqual(Set(before.nodeOrigins.values.map(\.instancePath)),Set(after.nodeOrigins.values.map(\.instancePath)))
    }
    func testNestedApplyCopiesAncestorsAndRejectsBrokenParentInterface()throws {
        let original=model(nested:true),rootID=original.source.root.instances[0].id,path=[rootID,"nested"]
        let begun=try ModelBlockEditing.beginDraft(original,path:path)
        var counter=0
        let result=try ModelBlockEditing.applyDraft(begun,path:path,expectedBaseRevision:4,idProvider:{counter+=1;return "copy-\(counter)"})
        XCTAssertEqual(counter,2);XCTAssertEqual(result.source.definitions.count,original.source.definitions.count+2)
        let outer=try XCTUnwrap(result.source.definitions.first{$0.reference == result.source.root.instances[0].definition})
        XCTAssertEqual(outer.body.instances[0].id,"nested");XCTAssertEqual(outer.body.instances[0].definition.id,"copy-1")
        XCTAssertEqual(Array(result.source.definitions.prefix(original.source.definitions.count)),original.source.definitions)
        var proposed=begun.drafts[0].proposed;proposed.outputs[0].id="renamed"
        let broken=try ModelBlockEditing.updateDraft(begun,proposed:proposed,expectedBaseRevision:4,path:path)
        XCTAssertThrowsError(try ModelBlockEditing.applyDraft(broken,path:path,expectedBaseRevision:4,idProvider:{UUID().uuidString}))
        XCTAssertEqual(original.revision,4)
    }
    func testDefinitionValidatorNestedCoverageCyclesAndExposedParameterCompatibility()throws {
        var source=model(nested:true).source
        let wrapper=source.definitions.last!
        XCTAssertNoThrow(try CompositeDefinitionValidation.validate(wrapper,definitions:source.definitions,registry:.builtIn()))
        var missing=wrapper;missing.inputs=[]
        XCTAssertThrowsError(try CompositeDefinitionValidation.validate(missing,definitions:source.definitions,registry:.builtIn()))
        var cyclic=wrapper;cyclic.body.instances[0].definition=wrapper.reference
        source.definitions[source.definitions.count-1]=cyclic
        XCTAssertThrowsError(try CompositeDefinitionValidation.validate(cyclic,definitions:source.definitions,registry:.builtIn()))
        var proposal=model().source.definitions[0]
        proposal.parameters[0].targets=[.init("missing","out_features")]
        XCTAssertThrowsError(try CompositeDefinitionValidation.validate(proposal,definitions:[proposal],registry:.builtIn()))
    }
    func testStandaloneDefinitionValidatorRejectsMalformedMetadataRecursively()throws {
        let source=model(nested:true).source,wrapper=source.definitions.last!
        var unsupported=wrapper;unsupported.body.configurationVersion=9
        XCTAssertThrowsError(try CompositeDefinitionValidation.validate(unsupported,definitions:source.definitions,registry:.builtIn()))
        for id in ["",String(repeating:"a",count:129)] {
            var malformed=wrapper;malformed.reference.id=id
            XCTAssertThrowsError(try CompositeDefinitionValidation.validate(malformed,definitions:source.definitions,registry:.builtIn()))
        }
        for version in [-1,Int(Int32.max)+1] {
            var malformed=wrapper;malformed.reference.version.major=version
            XCTAssertThrowsError(try CompositeDefinitionValidation.validate(malformed,definitions:source.definitions,registry:.builtIn()))
        }
        var nested=source.definitions
        nested[0].body.configurationVersion=9
        XCTAssertThrowsError(try CompositeDefinitionValidation.validate(wrapper,definitions:nested,registry:.builtIn()))
        var malformedChild=source.definitions[0];malformedChild.reference.version.patch = -1
        var parent=wrapper;parent.body.instances[0].definition=malformedChild.reference
        XCTAssertThrowsError(try CompositeDefinitionValidation.validate(parent,definitions:[malformedChild,parent],registry:.builtIn()))
        for version in [1,2] {
            var valid=wrapper;valid.body.configurationVersion=version
            XCTAssertNoThrow(try CompositeDefinitionValidation.validate(valid,definitions:source.definitions,registry:.builtIn()))
        }
    }
    func testRevisionIdentityAndCapacityFailuresAreAtomic()throws {
        let original=model(),path=[original.source.root.instances[0].id],begun=try ModelBlockEditing.beginDraft(original,path:path)
        XCTAssertThrowsError(try ModelBlockEditing.applyDraft(begun,path:path,expectedBaseRevision:4,idProvider:{original.source.definitions[0].reference.id}))
        var overflow=original;overflow.revision=UInt64.max
        overflow=try ModelBlockEditing.beginDraft(overflow,path:path)
        XCTAssertThrowsError(try ModelBlockEditing.applyDraft(overflow,path:path,expectedBaseRevision:UInt64.max,idProvider:{"copy"}))
        var full=original
        for index in 1..<64 {var definition=original.source.definitions[0];definition.reference.id="unused-\(index)";full.source.definitions.append(definition)}
        full=try ModelBlockEditing.beginDraft(full,path:path)
        XCTAssertThrowsError(try ModelBlockEditing.applyDraft(full,path:path,expectedBaseRevision:4,idProvider:{"copy"}))
        XCTAssertEqual(begun.source,original.source);XCTAssertEqual(full.source.definitions.count,64)
    }
}
