import XCTest
import KwipMLCore
@testable import KwipMLApp

final class Data02ObservationLedgerTests: XCTestCase {
    private let observedSHA = String(repeating:"a",count:64)
    private var identity: PreparationObservationIdentity { .init(requestID:"request-1",graphSHA256:observedSHA,sourceBindingsSHA256:String(repeating:"b",count:64),revision:"revision-1") }
    private var graph: DataFlowDocument { .init(nodes:[.init(id:"source",operatorID:"kwip.data.sourceCSV"),.init(id:"rename",operatorID:"kwip.data.columns",config:["rows":.array([.object(["source":.string("original"),"included":.bool(true),"outputName":.string("renamed"),"targetType":.string("keep")])]),"castFailure":.string("error"),"trimText":.bool(false),"newColumns":.string("drop"),"missingPolicy":.string("preserve")])],edges:[.init(from:.init(node:"source",port:"table"),to:.init(node:"rename",port:"table"))]) }
    private func ledger() throws -> PreparationObservationLedger { try .init(identity:identity,graph:graph,outputKinds:["source":["table":"table"],"rename":["table":"table"]],expectedSourceDigests:["source":observedSHA]) }
    private func preview(_ node:String="source",_ column:String="original") -> [String:JSONValue] {
        let value: [String:Any] = ["event":"dataflowPreview","nodeID":node,"execution":["nodeID":node,"status":"executed"],"outputs":["table":["kind":"table","lineage":["role":"unassigned","sourceDigest":observedSHA,"membershipDigest":observedSHA,"dataVersion":observedSHA,"partitionDigest":"","fittedStateDigests":[]],"preview":["partition":"unassigned","dataVersion":observedSHA,"rowCount":2,"columns":[["name":column,"dtype":"string","nullable":false]],"rows":[["a"],["b"]],"diagnostics":[],"diagnosticCount":0]]]]
        return try! JSONDecoder().decode([String:JSONValue].self,from:JSONSerialization.data(withJSONObject:value))
    }
    func testVersionedDeferredLiveIdentityDoesNotInventAContentHash() throws {
        var event=preview()
        guard case .object(var outputs)=event["outputs"],case .object(var table)=outputs["table"],
              case .object(var lineage)=table["lineage"],case .object(var sample)=table["preview"] else {return XCTFail("Fixture")}
        table["schemaVersion"] = 2;lineage["dataVersion"] = .null;sample["dataVersion"] = .null
        table["lineage"] = .object(lineage);table["preview"] = .object(sample)
        outputs["table"] = .object(table);event["outputs"] = .object(outputs)
        var value=try ledger();XCTAssertTrue(value.nodeStarted("source",identity:identity))
        XCTAssertTrue(value.consume(event,identity:identity))
        let observed=try XCTUnwrap(value.observation(at:.init(node:"source",port:"table"),current:identity))
        XCTAssertEqual(observed.rowCount,2);XCTAssertNil(observed.dataVersion)
        table.removeValue(forKey:"schemaVersion")
        XCTAssertThrowsError(try WorkflowPreparationPreviewValidation.validate(.object(table),kind:"table",operatorID:"kwip.data.sourceCSV"))
        table["schemaVersion"] = 3
        XCTAssertThrowsError(try WorkflowPreparationPreviewValidation.validate(.object(table),kind:"table",operatorID:"kwip.data.sourceCSV"))
        table["schemaVersion"] = 2;lineage["dataVersion"] = .string(observedSHA);table["lineage"] = .object(lineage)
        XCTAssertThrowsError(try WorkflowPreparationPreviewValidation.validate(.object(table),kind:"table",operatorID:"kwip.data.sourceCSV"))
    }

    func testReachedSchemaRetainsPartialFailureWithoutInventingDownstream() throws {
        var value=try ledger()
        XCTAssertTrue(value.nodeStarted("source",identity:identity))
        XCTAssertTrue(value.consume(preview(),identity:identity))
        value.finish(.failed)
        let observed=try XCTUnwrap(value.observation(at:.init(node:"source",port:"table"),current:identity))
        XCTAssertEqual(observed.columns.map(\.schema.name),["original"])
        XCTAssertEqual(value.status,.failed)
        XCTAssertTrue(value.disclosure(current:identity).contains("Partial"))
        XCTAssertNil(value.observation(at:.init(node:"rename",port:"table"),current:identity))
        XCTAssertFalse(value.consume(preview(),identity:identity),"Late frames cannot mutate retained evidence")
    }
    func testCancellationRetainsReachedColumnsAndSourceRevisionMakesThemStale() throws {
        var value=try ledger();_ = value.nodeStarted("source",identity:identity);_ = value.consume(preview(),identity:identity);value.finish(.cancelled)
        XCTAssertNotNil(value.observation(at:.init(node:"source",port:"table"),current:identity))
        let changed=PreparationObservationIdentity(requestID:identity.requestID,graphSHA256:observedSHA,sourceBindingsSHA256:String(repeating:"c",count:64),revision:identity.revision)
        XCTAssertNil(value.observation(at:.init(node:"source",port:"table"),current:changed))
        XCTAssertTrue(value.disclosure(current:changed).contains("Stale"))
        let revised=PreparationObservationIdentity(requestID:identity.requestID,graphSHA256:observedSHA,sourceBindingsSHA256:identity.sourceBindingsSHA256,revision:"revision-2")
        XCTAssertNil(value.observation(at:.init(node:"source",port:"table"),current:revised))
    }
    func testForeignUnstartedMalformedUnknownPortsAndDuplicateEventsAreAtomic() throws {
        var value=try ledger()
        XCTAssertFalse(value.consume(preview(),identity:identity))
        let foreign=PreparationObservationIdentity(requestID:"other",graphSHA256:observedSHA,sourceBindingsSHA256:identity.sourceBindingsSHA256,revision:identity.revision)
        XCTAssertFalse(value.nodeStarted("source",identity:foreign))
        XCTAssertTrue(value.nodeStarted("source",identity:identity))
        var malformed=preview();malformed["outputs"] = .object(["table":.object(["kind":.string("table")])])
        XCTAssertFalse(value.consume(malformed,identity:identity))
        XCTAssertNil(value.observation(at:.init(node:"source",port:"table"),current:identity))
        var unknown=preview();unknown["outputs"] = .object(["foreign":.object([:])])
        XCTAssertFalse(value.consume(unknown,identity:identity))
        XCTAssertFalse(value.consume(preview(),identity:foreign))
        XCTAssertTrue(value.consume(preview(),identity:identity))
        XCTAssertFalse(value.consume(preview(),identity:identity))
    }
    func testRenamesCarryOriginalColumnOriginAndImmediateObservedPort() throws {
        var value=try ledger();_ = value.nodeStarted("source",identity:identity);_ = value.consume(preview(),identity:identity)
        _ = value.nodeStarted("rename",identity:identity);XCTAssertTrue(value.consume(preview("rename","renamed"),identity:identity))
        let output=try XCTUnwrap(value.observation(at:.init(node:"rename",port:"table"),current:identity))
        XCTAssertEqual(output.columns.first?.origin.nodeID,"source")
        XCTAssertEqual(output.columns.first?.origin.columnName,"original")
        XCTAssertEqual(output.endpoint,.init(node:"rename",port:"table"))
        XCTAssertEqual(output.rowCount,2)
        XCTAssertEqual(output.sourceDigest,observedSHA)
    }
    func testCompletionRequiresExplicitVerifiedArchiveAndNeverPromotesFailure() throws {
        var value=try ledger();_ = value.nodeStarted("source",identity:identity);_ = value.consume(preview(),identity:identity)
        XCTAssertFalse(value.completeVerified(identity:identity,allOutputsVerified:false))
        XCTAssertEqual(value.status,.running)
        value.finish(.failed)
        XCTAssertFalse(value.completeVerified(identity:identity,allOutputsVerified:true))
        XCTAssertEqual(value.status,.failed)
        var accepted=try ledger();_ = accepted.nodeStarted("source",identity:identity);_ = accepted.consume(preview(),identity:identity)
        _ = accepted.nodeStarted("rename",identity:identity);_ = accepted.consume(preview("rename","renamed"),identity:identity)
        XCTAssertTrue(accepted.completeVerified(identity:identity,allOutputsVerified:true))
        XCTAssertEqual(accepted.status,.completed)
    }
    func testIdentityAndGraphBoundsRefuseInvalidConstruction() throws {
        let invalid=PreparationObservationIdentity(requestID:"request",graphSHA256:"not-a-observedSHA",sourceBindingsSHA256:observedSHA,revision:"r")
        XCTAssertThrowsError(try PreparationObservationLedger(identity:invalid,graph:graph,outputKinds:["source":["table":"table"]]))
        XCTAssertThrowsError(try PreparationObservationLedger(identity:identity,graph:graph,outputKinds:["source":["not a port":"table"],"rename":["table":"table"]]))
    }
    func testEmptyBranchRetainsItsSchemaWithoutInventingRows() throws {
        var value=try ledger();_ = value.nodeStarted("source",identity:identity)
        var event=preview()
        guard case .object(var outputs)=event["outputs"],case .object(var table)=outputs["table"],case .object(var sample)=table["preview"] else {return XCTFail("Bad test fixture")}
        sample["rowCount"]=0;sample["rows"] = .array([]);table["preview"] = .object(sample);outputs["table"] = .object(table);event["outputs"] = .object(outputs)
        XCTAssertTrue(value.consume(event,identity:identity))
        XCTAssertEqual(value.observation(at:.init(node:"source",port:"table"),current:identity)?.rowCount,0)
        var nonempty=sample;nonempty["rows"] = .array([.array([.string("impossible")])]);table["preview"] = .object(nonempty)
        XCTAssertThrowsError(try WorkflowPreparationPreviewValidation.validate(.object(table),kind:"table",operatorID:"kwip.data.sourceCSV"))
        var negative=sample;negative["rowCount"] = -1;table["preview"] = .object(negative)
        XCTAssertThrowsError(try WorkflowPreparationPreviewValidation.validate(.object(table),kind:"table",operatorID:"kwip.data.sourceCSV"))
    }

    func testLargeRowCountsUseCellBudgetAndRetainBoundedSamples() throws {
        guard case .object(let ports) = preview()["outputs"], case .object(let fixture) = ports["table"] else { return XCTFail("Bad fixture") }
        func table(rows: Int64, columns: Int) -> JSONValue {
            var value = fixture
            guard case .object(var sample) = value["preview"] else { return .null }
            sample["rowCount"] = .integer(rows)
            sample["columns"] = .array((0..<columns).map { .object(["name": .string("column_\($0)"), "dtype": .string("string"), "nullable": .bool(false)]) })
            sample["rows"] = .array([.array(Array(repeating: .string("sample"), count: columns))])
            value["preview"] = .object(sample)
            return .object(value)
        }
        XCTAssertNoThrow(try WorkflowPreparationPreviewValidation.validate(table(rows: 500_001, columns: 2), kind: "table", operatorID: "kwip.data.sourceCSV"))
        XCTAssertNoThrow(try WorkflowPreparationPreviewValidation.validate(table(rows: 2_500_000, columns: 2), kind: "table", operatorID: "kwip.data.sourceCSV"))
        XCTAssertThrowsError(try WorkflowPreparationPreviewValidation.validate(table(rows: 2_500_001, columns: 2), kind: "table", operatorID: "kwip.data.sourceCSV"))
        XCTAssertNoThrow(try WorkflowPreparationPreviewValidation.validate(table(rows: 39_062, columns: 128), kind: "table", operatorID: "kwip.data.sourceCSV"))
        XCTAssertThrowsError(try WorkflowPreparationPreviewValidation.validate(table(rows: 39_063, columns: 128), kind: "table", operatorID: "kwip.data.sourceCSV"))
        XCTAssertThrowsError(try WorkflowPreparationPreviewValidation.validate(table(rows: Int64.max, columns: 1), kind: "table", operatorID: "kwip.data.sourceCSV"))
    }

    func testForeignSourceLineageCannotBecomeObservedEvenWithCurrentCallback() throws {
        var value=try ledger();_ = value.nodeStarted("source",identity:identity)
        var event=preview()
        guard case .object(var outputs)=event["outputs"],case .object(var table)=outputs["table"],case .object(var lineage)=table["lineage"] else {return XCTFail("Bad fixture")}
        lineage["sourceDigest"] = .string(String(repeating:"d",count:64));table["lineage"] = .object(lineage);outputs["table"] = .object(table);event["outputs"] = .object(outputs)
        XCTAssertFalse(value.consume(event,identity:identity))
        XCTAssertNil(value.observation(at:.init(node:"source",port:"table"),current:identity))
        XCTAssertTrue(value.consume(preview(),identity:identity))
        _ = value.nodeStarted("rename",identity:identity)
        event=preview("rename","renamed")
        guard case .object(var later)=event["outputs"],case .object(var renamed)=later["table"],case .object(var nextLineage)=renamed["lineage"] else {return XCTFail("Bad fixture")}
        nextLineage["sourceDigest"] = .string(String(repeating:"d",count:64));renamed["lineage"] = .object(nextLineage);later["table"] = .object(renamed);event["outputs"] = .object(later)
        XCTAssertFalse(value.consume(event,identity:identity))
    }

    private func withSource(_ event:[String:JSONValue],_ digest:String)->[String:JSONValue] {
        var event=event
        guard case .object(var outputs)=event["outputs"],case .object(var table)=outputs["table"],case .object(var lineage)=table["lineage"] else {return event}
        lineage["sourceDigest"] = .string(digest);table["lineage"] = .object(lineage);outputs["table"] = .object(table);event["outputs"] = .object(outputs)
        return event
    }
    func testUnionRequiresReachedOrderedParentsAndDerivedSourceDigest() throws {
        let graph=DataFlowDocument(nodes:[.init(id:"source",operatorID:"kwip.data.sourceCSV"),.init(id:"right",operatorID:"kwip.data.sourceCSV"),.init(id:"union",operatorID:"kwip.data.union",config:UnionConfiguration.defaults)],edges:[.init(from:.init(node:"source",port:"table"),to:.init(node:"union",port:"table1")),.init(from:.init(node:"right",port:"table"),to:.init(node:"union",port:"table2"))])
        var value=try PreparationObservationLedger(identity:identity,graph:graph,outputKinds:["source":["table":"table"],"right":["table":"table"],"union":["table":"table"]],expectedSourceDigests:["source":observedSHA,"right":String(repeating:"b",count:64)])
        _ = value.nodeStarted("source",identity:identity);_ = value.consume(preview(),identity:identity)
        _ = value.nodeStarted("union",identity:identity)
        let union=withSource(preview("union"),"58178f099b5e5527f3e429e92ff970c0b9c2f746c8bb672d97092bc8109e1d54")
        XCTAssertFalse(value.consume(union,identity:identity))
        _ = value.nodeStarted("right",identity:identity);XCTAssertTrue(value.consume(withSource(preview("right"),String(repeating:"b",count:64)),identity:identity))
        XCTAssertFalse(value.consume(preview("union"),identity:identity))
        XCTAssertTrue(value.consume(union,identity:identity))
    }
    func testJoinedSourcesRequireBothReachedParentsAndDerivedSourceDigest() throws {
        let graph=DataFlowDocument(nodes:[.init(id:"source",operatorID:"kwip.data.sourceCSV"),.init(id:"right",operatorID:"kwip.data.sourceCSV"),.init(id:"join",operatorID:"kwip.data.join")],edges:[.init(from:.init(node:"source",port:"table"),to:.init(node:"join",port:"left")),.init(from:.init(node:"right",port:"table"),to:.init(node:"join",port:"right"))])
        var value=try PreparationObservationLedger(identity:identity,graph:graph,outputKinds:["source":["table":"table"],"right":["table":"table"],"join":["table":"table"]],expectedSourceDigests:["source":observedSHA,"right":String(repeating:"b",count:64)])
        _ = value.nodeStarted("source",identity:identity);_ = value.consume(preview(),identity:identity)
        _ = value.nodeStarted("join",identity:identity)
        let joined=withSource(preview("join"),"6bac6ae7b1f7167811b0ae97961425e36e39e7c27005591f4c570e4d6ec2ab46")
        XCTAssertFalse(value.consume(joined,identity:identity),"An unvisited parent cannot supply observed lineage")
        _ = value.nodeStarted("right",identity:identity);XCTAssertTrue(value.consume(withSource(preview("right"),String(repeating:"b",count:64)),identity:identity))
        XCTAssertFalse(value.consume(preview("join"),identity:identity),"A source identity cannot stand in for the joined identity")
        XCTAssertTrue(value.consume(joined,identity:identity))
    }

    func testAggregateSourceIdentityUsesActualGroupNamesIncludingUnicode() throws {
        let graph=DataFlowDocument(nodes:[.init(id:"source",operatorID:"kwip.data.sourceCSV"),.init(id:"aggregate",operatorID:"kwip.data.aggregate",config:["groupBy":.array([.string("café🐕")]),"nullKeys":.string("drop")])],edges:[.init(from:.init(node:"source",port:"table"),to:.init(node:"aggregate",port:"table"))])
        var value=try PreparationObservationLedger(identity:identity,graph:graph,outputKinds:["source":["table":"table"],"aggregate":["table":"table"]],expectedSourceDigests:["source":observedSHA])
        _ = value.nodeStarted("source",identity:identity);_ = value.consume(preview("source","café🐕"),identity:identity)
        _ = value.nodeStarted("aggregate",identity:identity)
        XCTAssertFalse(value.consume(preview("aggregate"),identity:identity))
        let expected="59d82662ca0976cfabf5ae6b74eebdc000ba43c5100bbfae8db8814706f02e0f"
        XCTAssertTrue(value.consume(withSource(preview("aggregate","café🐕"),expected),identity:identity),"Expected digest independently computed with hashlib and compact ensure_ascii JSON")
    }

}
