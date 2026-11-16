import XCTest
import KwipMLCore
@testable import KwipMLApp

final class MCPMetadataReadTests:XCTestCase {
    func page(_ source:[String:JSONValue],_ arguments:[String:JSONValue]=[:],budget:Int=16000)throws->[String:JSONValue] {
        let result=try MCPMetadataRead.response(source,arguments:arguments,byteLimit:budget)
        XCTAssertLessThanOrEqual(try JSONEncoder().encode(result).count,budget)
        guard case .object(let page)=result["metadataRead"] else {throw MCPSettings.Refusal(code:"missing_page")};return page
    }
    func testSmallMetadataRetainsLegacyShapeAndPrivateKeysCannotBeTraversed()throws {
        let source:[String:JSONValue] = ["projectID":.string("exact-id"),"password":.string("private")]
        XCTAssertEqual(try MCPMetadataRead.response(source,arguments:[:],byteLimit:1000),["projectID":.string("exact-id")])
        XCTAssertThrowsError(try page(source,["readPath":.array([.string("password")])]))
    }
    func testLargeChildrenAreReferencesAndUnicodeTextReassemblesWithoutLoss()throws {
        let text=String(repeating:"é👩🏽‍💻\n",count:12000),source:[String:JSONValue] = ["lesson":.object(["id":.string("exact-lesson-id"),"body":.string(text)])]
        let root=try page(source)
        guard case .array(let entries)=root["entries"],case .object(let reference)=entries.first else {return XCTFail("Missing root reference")}
        XCTAssertEqual(reference["complete"],false);XCTAssertNil(reference["value"])
        var arguments:[String:JSONValue] = ["readPath":.array(["lesson","body"].map(JSONValue.string)),"readRevision":root["readRevision"]!,"readLimit":4096]
        var reconstructed="",pages=0
        repeat {
            let result=try page(source,arguments);reconstructed += try XCTUnwrap(result["value"]?.stringValue);pages+=1
            XCTAssertEqual(result["offsetUnit"],.string("unicodeScalar"))
            if case .object(let next)=result["nextRead"] {arguments=next} else {break}
        } while pages<100
        XCTAssertEqual(reconstructed,text);XCTAssertGreaterThan(pages,1)
    }
    func testDeepConfigurationUsesReferencesInsteadOfDepthPlaceholder()throws {
        var nested:JSONValue = .string("exact-leaf")
        for _ in 0..<40 {nested = .object(["child":nested])}
        let source:[String:JSONValue] = ["configuration":nested]
        let root=try page(source)
        guard case .array(let entries)=root["entries"],case .object(let entry)=entries.first else {return XCTFail("Missing reference")}
        XCTAssertEqual(entry["complete"],false)
        let leaf=try page(source,["readPath":.array((["configuration"]+Array(repeating:"child",count:40)).map(JSONValue.string)),"readRevision":root["readRevision"]!])
        XCTAssertEqual(leaf["value"],.string("exact-leaf"))
    }
    func testCollectionPaginationDoesNotSkipLargeNodeAndRejectsChangedSource()throws {
        let nodes=(0..<205).map{JSONValue.object(["id":.string("node-\($0)"),"configuration":.string(String(repeating:"x",count:6000))])}
        let source:[String:JSONValue] = ["nodes":.array(nodes)]
        var arguments:[String:JSONValue] = ["readPath":.array([.string("nodes")]),"readLimit":33]
        var keys:[String]=[],revision:JSONValue?
        repeat {
            let result=try page(source,arguments);revision=result["readRevision"]
            guard case .array(let entries)=result["entries"] else {return XCTFail("Missing entries")}
            keys += entries.compactMap{if case .object(let row)=$0{return row["key"]?.stringValue};return nil}
            for entry in entries {guard case .object(let row)=entry else {continue};XCTAssertEqual(row["complete"],false)}
            if case .object(let next)=result["nextRead"] {arguments=next} else {break}
        } while keys.count<300
        XCTAssertEqual(keys,(0..<205).map(String.init))
        let node=try page(source,["readPath":.array(["nodes","204","id"].map(JSONValue.string)),"readRevision":revision!])
        XCTAssertEqual(node["value"],.string("node-204"))
        XCTAssertThrowsError(try page(["nodes":.array(nodes+[.string("changed")])],["readPath":.array([.string("nodes")]),"readRevision":revision!])) {error in
            XCTAssertEqual((error as? MCPSettings.Refusal)?.code,"metadata_read_stale")
        }
        XCTAssertThrowsError(try page(source,["readPath":.array([.string("nodes")]),"readOffset":1]))
    }
}

extension MCPWorkflowGatewayTests {
    /// Existing schema assertions now request only the component details they use.
    func catalogWithDetails(_ f:Fixture,families:Set<String>)async throws->[String:JSONValue] {
        var catalog=try await call(f,.catalogList)
        guard case .array(let rows)=catalog["components"] else {throw MCPSettings.Refusal(code:"missing_catalog")}
        var expanded:[JSONValue]=[]
        for value in rows {
            guard case .object(let row)=value,let family=row["familyID"]?.stringValue,families.contains(family),let id=row["componentID"] else {expanded.append(value);continue}
            let detail=try await call(f,.catalogGet,["componentID":id])
            let complete=try await completeMetadata(f,action:.catalogGet,arguments:["componentID":id],response:detail)
            guard case .object(let result)=complete,case .array(let items)=result["components"],let item=items.first else {throw MCPSettings.Refusal(code:"missing_component")}
            expanded.append(item);catalog["patchOperationSchema"]=result["patchOperationSchema"]
        }
        catalog["components"] = .array(expanded);return catalog
    }
    private func completeMetadata(_ f:Fixture,action:MCPAction,arguments:[String:JSONValue],response:[String:JSONValue])async throws->JSONValue {
        guard case .object(var page)=response["metadataRead"] else {return .object(response)}
        let kind=page["kind"]?.stringValue
        if let value=page["value"],kind != "string" {return value}
        var fields:[String:JSONValue]=[:],items:[JSONValue]=[],text=""
        while true {
            if kind == "string" {text += page["value"]?.stringValue ?? ""}
            if case .array(let entries)=page["entries"] {
                for entry in entries {
                    guard case .object(let row)=entry,let key=row["key"]?.stringValue else {continue}
                    let value:JSONValue
                    if let complete=row["value"] {value=complete}
                    else {
                        var child=arguments;child["readPath"]=row["readPath"];child["readRevision"]=page["readRevision"]
                        value=try await completeMetadata(f,action:action,arguments:child,response:try await call(f,action,child))
                    }
                    if kind == "object" {fields[key]=value} else {items.append(value)}
                }
            }
            guard case .object(let next)=page["nextRead"] else {break}
            let request=arguments.merging(next){_,new in new}
            let result=try await call(f,action,request)
            guard case .object(let nextPage)=result["metadataRead"] else {throw MCPSettings.Refusal(code:"missing_page")};page=nextPage
        }
        if kind == "object" {return .object(fields)}
        if kind == "array" {return .array(items)}
        return .string(text)
    }
}
