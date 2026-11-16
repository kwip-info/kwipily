import XCTest
import CryptoKit
@testable import KwipMLCore

final class MediaArtifactReferenceTests:XCTestCase {
    func fixture()throws->(Data,Data) {
        let header:[String:Any]=["schemaVersion":1,"kind":"audioSamples","shape":[2,1],"axes":["samples","channels"],"dtype":"int16","metadata":["sampleRate":48000,"sourceBits":16],"byteOrder":"little"]
        let bytes=try JSONSerialization.data(withJSONObject:header,options:[.sortedKeys,.withoutEscapingSlashes])
        var container=Data("KWIPMEDIA1\n".utf8);let size=UInt32(bytes.count)
        container.append(contentsOf:[UInt8((size>>24)&255),UInt8((size>>16)&255),UInt8((size>>8)&255),UInt8(size&255)])
        container.append(bytes);container.append(contentsOf:[0,128,255,127])
        var ref=header;ref.removeValue(forKey:"byteOrder");ref["kind"]="audioSamplesRef"
        ref["sha256"]=SHA256.hash(data:container).map{String(format:"%02x",$0)}.joined()
        return (container,try JSONSerialization.data(withJSONObject:ref,options:[.sortedKeys]))
    }
    func testExactPinnedPCMContainerAndReference()throws {
        let(container,raw)=try fixture();let ref=try MediaArtifactReference.decode(raw)
        XCTAssertEqual(ref.shape,[2,1]);XCTAssertEqual(ref.axes,["samples","channels"])
        XCTAssertEqual(try MediaArtifactReference.decode(container:container,expected:ref).samples,Data([0,128,255,127]))
        var changed=container;changed[changed.count-1]^=1
        XCTAssertThrowsError(try MediaArtifactReference.decode(container:changed,expected:ref))
    }
    func testPathsUnknownFieldsAxesAndPrecisionAreRejected()throws {
        let(_,raw)=try fixture();let original=try XCTUnwrap(JSONSerialization.jsonObject(with:raw) as? [String:Any])
        for (key,value) in [("path","/private/file" as Any),("axes",["channels","samples"] as Any),("dtype","float32" as Any),("shape",[Int.max,Int.max] as Any)] {
            var bad=original;bad[key]=value
            XCTAssertThrowsError(try MediaArtifactReference.decode(JSONSerialization.data(withJSONObject:bad)))
        }
    }
}

extension MediaArtifactReferenceTests {
    func testPreparedPlanV3AndFloatingModelInputShapePreserveLegacyFeatures()throws {
        let registry=try ComponentRegistry.builtIn()
        let values:[String:JSONValue]=["schemaVersion":3,"maximumBatchBytes":1048576,"inputs":.array([.object(["name":.string("image"),"nodeID":.string("pixels"),"representation":.string("prepared"),"dtype":.string("float32"),"sampleShape":.array([1,2,2]),"sampleAxes":.array(["channels","height","width"].map(JSONValue.string)),"columns":.array([.string("prepared")])])]),"roles":.array([])]
        let plan=try NamedInputPlan(values:values);XCTAssertEqual(plan.inputs[0].sampleAxes,["channels","height","width"])
        let tensor=try FloatingInputConfiguration(["shape":.array([1,2,2])]).tensor
        XCTAssertEqual(tensor.layout.sample.dimensions,[1,2,2]);XCTAssertEqual(tensor.dtype,.float32)
        let node=GraphNode(id:"input",type:"input",config:["shape":.array([1,2,2])])
        XCTAssertEqual(try GraphMigration.effectiveConfiguration(of:node,schemaVersion:2,registry:registry),node.config)
        XCTAssertThrowsError(try GraphMigration.effectiveConfiguration(of:node,schemaVersion:1,registry:registry))
        let old=GraphNode(id:"old",type:"input",config:["features":2])
        XCTAssertEqual(try GraphMigration.effectiveConfiguration(of:old,schemaVersion:2,registry:registry),old.config)
        XCTAssertEqual(registry.component(type:"input")?.fields(configuration:node.config).map(\.key),["shape"])
    }
    func testMediaCatalogValidationRejectsUnboundedAndCrossKindDrafts()throws {
        var values=try PrepareMediaConfiguration.defaults(kind:"audio")
        try PrepareMediaConfiguration.validate(values)
        values["hopSamples"] = 4
        XCTAssertThrowsError(try PrepareMediaConfiguration.validate(values))
        values["windowSamples"] = 8
        try PrepareMediaConfiguration.validate(values)
        values["sizeMode"] = .string("resize")
        XCTAssertThrowsError(try PrepareMediaConfiguration.validate(values))
    }
}

extension MediaArtifactReferenceTests {
    func testPythonPreparedGeometryContainersDecodeWithoutNumericDrift()throws {
        guard let python=ProcessInfo.processInfo.environment["KWIPML_BACKEND_TEST_PYTHON"] else {throw XCTSkip("Set pinned Python for media-container parity.")}
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let process=Process(),output=Pipe(),errors=Pipe()
        process.executableURL=URL(fileURLWithPath:python);process.currentDirectoryURL=root
        process.arguments=["-c","""
import base64,json
from runner.media_artifacts import encode_asset
rows=[]
for scale in [1.,.5,2/3,1/3,1/16384,16384.,.0001,1.23456789012345]:
    raw,ref=encode_asset(bytes(4),kind='tensor',shape=[1],axes=['features'],dtype='float32',metadata={'geometry':[scale,0.,-.25,0.,scale,-.75,0.,0.,1.]})
    rows.append({'container':base64.b64encode(raw).decode(),'reference':ref})
print(json.dumps(rows))
"""]
        process.standardOutput=output;process.standardError=errors
        try process.run();let bytes=output.fileHandleForReading.readDataToEndOfFile();process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus,0,String(decoding:errors.fileHandleForReading.readDataToEndOfFile(),as:UTF8.self))
        struct Row:Decodable {let container:String;let reference:MediaArtifactReference}
        let rows=try JSONDecoder().decode([Row].self,from:bytes);XCTAssertEqual(rows.count,8)
        for row in rows {_=try MediaArtifactReference.decode(container:try XCTUnwrap(Data(base64Encoded:row.container)),expected:row.reference)}
    }
}

extension MediaArtifactReferenceTests {
    func testPreparedPCMPrecisionIsAdditiveAndRejectsMismatchedTypes()throws {
        for (bits,dtype) in [(8,"uint8"),(16,"int16"),(24,"int32"),(32,"int32")] {
            var metadata:[String:Any] = ["sampleRate":16000,"sourceBits":bits,"validSamples":2,"sourceStart":0,"sourceEnd":2,"sourceRate":16000]
            var fields:[String:Any] = ["schemaVersion":1,"kind":"preparedTensorRef","sha256":String(repeating:"a",count:64),"shape":[1,4],"axes":["channels","samples"],"dtype":dtype,"metadata":metadata]
            XCTAssertEqual(try MediaArtifactReference.decode(JSONSerialization.data(withJSONObject:fields)).metadata["sourceBits"],.integer(Int64(bits)))
            metadata["sourceBits"]=12;fields["metadata"]=metadata
            XCTAssertThrowsError(try MediaArtifactReference.decode(JSONSerialization.data(withJSONObject:fields)))
            metadata.removeValue(forKey:"sourceBits");fields["metadata"]=metadata
            XCTAssertNoThrow(try MediaArtifactReference.decode(JSONSerialization.data(withJSONObject:fields)))
            metadata["sourceBits"]=bits;metadata["maskConvention"]="valid"
            fields["metadata"]=metadata;fields["shape"]=[4];fields["axes"]=["samples"];fields["dtype"]="int64"
            XCTAssertNoThrow(try MediaArtifactReference.decode(JSONSerialization.data(withJSONObject:fields)))
        }
    }
}
