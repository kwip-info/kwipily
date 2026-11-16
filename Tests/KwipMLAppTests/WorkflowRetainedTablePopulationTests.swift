import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowRetainedTablePopulationTests:XCTestCase {
    private struct Fixture {
        let root:URL,pack:URL,envelope:Data,file:WorkflowArchiveFile,ids:[String]
        func reader(inventory:[WorkflowArchiveFile]?=nil)throws->WorkflowRetainedTablePopulation {
            try .init(root:root,prefix:["preparation","artifacts"],inventory:inventory ?? [file],maximumBytes:file.bytes)
        }
    }
    private func fixture(selected:Bool=false,internalIndex:Bool=false,duplicateIDs:Bool=false,badSpan:Bool=false,badPosition:Bool=false)throws->Fixture {
        let root=try canonicalBackendTemporaryDirectory(),folder=root.appendingPathComponent("preparation/artifacts")
        try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true)
        var bytes=Data("KWIP-TABLE-PACK".utf8)+Data([0,1,10])
        func add(_ value:JSONValue)throws->JSONValue {
            let raw=try JSONEncoder().encode(value),offset=bytes.count;bytes.append(raw)
            return .object(["pack":.null,"object":.object(["offset":.integer(Int64(offset)),"bytes":.integer(Int64(raw.count)),"sha256":.string(WorkflowArchiveCodec.hash(raw))])])
        }
        let original=duplicateIDs ? ["source:0","source:0","source:2"]:["source:0","source:1","source:2"]
        let source=try add(.array(original.map(JSONValue.string)))
        let ids=selected ? [original[2],original[0]]:original
        let identity:JSONValue
        if selected {
            identity=try add(.object(["schemaVersion":1,"kind":.string("selectedColumn"),"source":source,"sourceCount":3,"positions":.array([badPosition ? 3:2,0])]))
        } else {identity=source}
        func index(_ reference:JSONValue,count:Int,height:Int=0)throws->JSONValue {
            let value=try add(.object(["schemaVersion":1,"kind":.string("orderedIndex"),"height":.integer(Int64(height)),"count":.integer(Int64(count)),
                "entries":.array([.object(["count":.integer(Int64(count+(badSpan ? 1:0))),height==0 ? "value":"reference":reference])])]))
            return .object(["reference":value,"count":.integer(Int64(count)),"height":.integer(Int64(height))])
        }
        let leaf=try index(identity,count:ids.count)
        let identityRoot:JSONValue
        if internalIndex,case .object(let values)=leaf {identityRoot=try index(values["reference"]!,count:ids.count,height:1)}else{identityRoot=leaf}
        let numbers=try add(.array(ids.indices.map{.integer(Int64($0))})),column=try index(numbers,count:ids.count)
        let columns:JSONValue = .array([.object(["name":.string("signal"),"dtype":.string("float64"),"nullable":false])])
        var lineage:[String:JSONValue]=["role":.string("test"),"fittedStateDigests":.array([]),"diagnostics":.array([])]
        for name in ["sourceDigest","membershipDigest","dataVersion","partitionDigest"] {lineage[name] = .string(String(repeating:"a",count:64))}
        var table:[String:JSONValue]=["schemaVersion":selected ? 2:1,"kind":.string("columnTable"),"columns":columns,"rowCount":.integer(Int64(ids.count)),"columnIndexes":.array([column]),"identityIndex":identityRoot,"lineage":.object(lineage)]
        if selected {table["selectionDepth"]=1}
        let reference=try add(.object(table)),name="table-pack-"+String(repeating:"b",count:32)+".kwipdata"
        let file=WorkflowArchiveFile(path:"preparation/artifacts/"+name,bytes:bytes.count,sha256:WorkflowArchiveCodec.hash(bytes))
        let pack=folder.appendingPathComponent(name);try bytes.write(to:pack)
        guard case .object(var rooted)=reference else {throw WorkflowArchiveError("Fixture reference")}
        rooted["pack"] = .object(["schemaVersion":1,"kind":.string("tableArtifactPack"),"path":.string(name),"bytes":.integer(Int64(file.bytes)),"sha256":.string(file.sha256)])
        let envelope=try JSONEncoder().encode(JSONValue.object(["schemaVersion":4,"kind":.string("segmentedTable"),"columns":columns,"table":.object(rooted)]))
        return .init(root:root,pack:pack,envelope:envelope,file:file,ids:ids)
    }
    func testPinnedPopulationAndInternalIndexWithoutRowMaterialization()throws {
        for nested in [false,true] {
            let f=try fixture(internalIndex:nested);defer{try? FileManager.default.removeItem(at:f.root)}
            let reader=try f.reader(),lineage=try reader.lineage(f.envelope)
            XCTAssertEqual(lineage["observationIDs"] as? [String],f.ids)
            XCTAssertEqual(lineage["role"] as? String,"test");try reader.finish()
        }
    }
    func testPinnedSelectionKeepsDeclaredOrder()throws {
        let f=try fixture(selected:true,internalIndex:true);defer{try? FileManager.default.removeItem(at:f.root)}
        let reader=try f.reader()
        XCTAssertEqual(try reader.lineage(f.envelope)["observationIDs"] as? [String],["source:2","source:0"])
        try reader.finish()
    }
    func testDuplicatePopulationBadIndexSpanAndOutOfRangeSelectionRefused()throws {
        for mode in 0..<3 {
            let f=try fixture(selected:mode==2,duplicateIDs:mode==0,badSpan:mode==1,badPosition:mode==2)
            defer{try? FileManager.default.removeItem(at:f.root)}
            XCTAssertThrowsError(try f.reader().lineage(f.envelope))
        }
    }
    func testMissingConflictingAndOutOfBoundsReferencesRefused()throws {
        let f=try fixture();defer{try? FileManager.default.removeItem(at:f.root)}
        XCTAssertThrowsError(try f.reader(inventory:[]).lineage(f.envelope))
        var value=try JSONDecoder().decode([String:JSONValue].self,from:f.envelope)
        guard case .object(let originalReference)=value["table"],case .object(var range)=originalReference["object"],case .object(let descriptor)=originalReference["pack"] else {return XCTFail("Fixture")}
        var reference=originalReference
        range["offset"]=0;reference["object"] = .object(range);value["table"] = .object(reference)
        XCTAssertThrowsError(try f.reader().lineage(JSONEncoder().encode(value)))
        reference=originalReference
        var changed=descriptor;changed["sha256"] = .string(String(repeating:"f",count:64));reference["pack"] = .object(changed);value["table"] = .object(reference)
        XCTAssertThrowsError(try f.reader().lineage(JSONEncoder().encode(value)))
    }
    func testWholeSegmentTamperAndReplacementAfterReadRefused()throws {
        let f=try fixture();defer{try? FileManager.default.removeItem(at:f.root)}
        let original=try Data(contentsOf:f.pack),reader=try f.reader()
        _=try reader.lineage(f.envelope)
        let numeric=try XCTUnwrap(original.range(of:Data("[0,1,2]".utf8)))
        var tampered=original;tampered[numeric.lowerBound+1] = 57;try tampered.write(to:f.pack)
        XCTAssertThrowsError(try reader.finish())
        XCTAssertThrowsError(try f.reader().lineage(f.envelope))
        try original.write(to:f.pack,options:.atomic)
        XCTAssertThrowsError(try reader.lineage(f.envelope))
        XCTAssertNoThrow(try f.reader().lineage(f.envelope))
    }
    func testSymlinkDuplicateJSONAndAggregateBudgetRefused()throws {
        let f=try fixture();defer{try? FileManager.default.removeItem(at:f.root)}
        XCTAssertThrowsError(try WorkflowRetainedTablePopulation(root:f.root,prefix:["preparation","artifacts"],inventory:[f.file],maximumBytes:f.file.bytes-1))
        let duplicate=Data("{\"schemaVersion\":4,\"schemaVersion\":4,\"kind\":\"segmentedTable\"}".utf8)
        XCTAssertThrowsError(try f.reader().lineage(duplicate))
        let moved=f.pack.appendingPathExtension("original");try FileManager.default.moveItem(at:f.pack,to:moved)
        try FileManager.default.createSymbolicLink(at:f.pack,withDestinationURL:moved)
        XCTAssertThrowsError(try f.reader().lineage(f.envelope))
    }
    func testLegacyFlatPopulationRemainsReadable()throws {
        let f=try fixture();defer{try? FileManager.default.removeItem(at:f.root)}
        let reader=try f.reader(),lineage=try reader.lineage(f.envelope)
        let value:[String:Any]=["schemaVersion":1,"kind":"table","columns":[["name":"signal","dtype":"float64","nullable":false]],"rows":[[1.0],[2.0],[3.0]],"lineage":lineage]
        XCTAssertEqual(try reader.lineage(JSONSerialization.data(withJSONObject:value))["observationIDs"] as? [String],f.ids)
    }
    func testParentSegmentReferencesRetainTheirOwnerContext()throws {
        let f=try fixture();defer{try? FileManager.default.removeItem(at:f.root)}
        var envelope=try JSONDecoder().decode([String:JSONValue].self,from:f.envelope)
        guard case .object(let reference)=envelope["table"],case .object(let range)=reference["object"],
              let offset=range["offset"]?.integerValue,let length=range["bytes"]?.integerValue else {return XCTFail("Fixture")}
        let old=try Data(contentsOf:f.pack)
        var metadata=try JSONDecoder().decode([String:JSONValue].self,from:old.subdata(in:offset..<offset+length))
        func rooted(_ value:JSONValue)throws->JSONValue {
            guard case .object(var index)=value,case .object(var entry)=index["reference"] else {throw WorkflowArchiveError("Fixture")}
            entry["pack"]=reference["pack"];index["reference"] = .object(entry);return .object(index)
        }
        guard case .array(let columns)=metadata["columnIndexes"] else {return XCTFail("Fixture")}
        metadata["columnIndexes"] = .array(try columns.map(rooted));metadata["identityIndex"] = try rooted(metadata["identityIndex"]!)
        var bytes=Data("KWIP-TABLE-PACK".utf8)+Data([0,1,10]);let start=bytes.count,raw=try JSONEncoder().encode(metadata);bytes.append(raw)
        let name="table-pack-"+String(repeating:"c",count:32)+".kwipdata",file=WorkflowArchiveFile(path:"preparation/artifacts/"+name,bytes:bytes.count,sha256:WorkflowArchiveCodec.hash(bytes))
        try bytes.write(to:f.pack.deletingLastPathComponent().appendingPathComponent(name))
        envelope["table"] = .object(["pack":.object(["schemaVersion":1,"kind":.string("tableArtifactPack"),"path":.string(name),"bytes":.integer(Int64(file.bytes)),"sha256":.string(file.sha256)]),
            "object":.object(["offset":.integer(Int64(start)),"bytes":.integer(Int64(raw.count)),"sha256":.string(WorkflowArchiveCodec.hash(raw))])])
        let reader=try WorkflowRetainedTablePopulation(root:f.root,prefix:["preparation","artifacts"],inventory:[f.file,file],maximumBytes:f.file.bytes+file.bytes)
        XCTAssertEqual(try reader.lineage(JSONEncoder().encode(envelope))["observationIDs"] as? [String],f.ids)
        try reader.finish()
    }
}
