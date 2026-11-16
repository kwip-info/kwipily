import XCTest
import CryptoKit
@testable import KwipMLCore

final class FileImageReferenceTests:XCTestCase {
    private func fixture(mode:String="RGB",dtype:String="uint8",channels:Int=3,layout:String="HWC")throws->(Data,FileImageReference) {
        let sizes=["uint8":1,"uint16":2,"int32":4,"float32":4]
        let header:[String:Any]=["schemaVersion":1,"kind":"imagePixels","width":2,"height":1,"channels":channels,"dtype":dtype,"layout":layout,"byteOrder":"little","policy":["orientation":"stored","colorMode":"preserve","palette":"expandRGBA","actualMode":mode]]
        let encoded=try JSONSerialization.data(withJSONObject:header,options:[.sortedKeys,.withoutEscapingSlashes])
        var data=Data("KWIPIMG1\n".utf8);let size=UInt32(encoded.count)
        data.append(contentsOf:[UInt8((size>>24)&255),UInt8((size>>16)&255),UInt8((size>>8)&255),UInt8(size&255)])
        data.append(encoded);data.append(Data(repeating:0,count:2*channels*sizes[dtype]!))
        let pin=SHA256.hash(data:data).map {String(format:"%02x",$0)}.joined()
        return (data,try .init(sha256:pin,width:2,height:1,channels:channels,dtype:dtype,layout:layout))
    }
    func testAllPreservedPixelModesAndLayoutsDecodeUnderTheirExactTypedPin()throws {
        for (mode,dtype,channels) in [("L","uint8",1),("LA","uint8",2),("RGB","uint8",3),("RGBA","uint8",4),("CMYK","uint8",4),("I;16","uint16",1),("I","int32",1),("F","float32",1)] {
            for layout in ["HWC","CHW"] {
                let(data,reference)=try fixture(mode:mode,dtype:dtype,channels:channels,layout:layout)
                let decoded=try FileImageReference.decode(container:data,expected:reference)
                XCTAssertEqual(decoded.reference,reference);XCTAssertEqual(decoded.policy["actualMode"],mode)
                XCTAssertEqual(try FileImageReference.decode(JSONEncoder().encode(reference)),reference)
            }
        }
    }
    func testReferencesDoNotGrantPathsOrAcceptUnknownFieldsOrInvalidGeometry()throws {
        let(_,reference)=try fixture()
        var value=try XCTUnwrap(JSONSerialization.jsonObject(with:JSONEncoder().encode(reference)) as? [String:Any])
        value["path"]="/private/ungranted-image"
        XCTAssertThrowsError(try FileImageReference.decode(JSONSerialization.data(withJSONObject:value)))
        value.removeValue(forKey:"path");value["width"]=true
        XCTAssertThrowsError(try FileImageReference.decode(JSONSerialization.data(withJSONObject:value)))
        XCTAssertThrowsError(try FileImageReference(sha256:reference.sha256,width:16385,height:1,channels:3,dtype:"uint8",layout:"HWC"))
    }
    func testChangedBytesGeometryAndColorInterpretationCannotReuseAnAssetPin()throws {
        let(data,reference)=try fixture()
        var changed=data;changed[changed.count-1]=1
        XCTAssertThrowsError(try FileImageReference.decode(container:changed,expected:reference))
        let wrong=try FileImageReference(sha256:reference.sha256,width:1,height:2,channels:3,dtype:"uint8",layout:"HWC")
        XCTAssertThrowsError(try FileImageReference.decode(container:data,expected:wrong))
        let(_,rgba)=try fixture(mode:"RGBA",channels:4),(_,cmyk)=try fixture(mode:"CMYK",channels:4)
        XCTAssertNotEqual(rgba.sha256,cmyk.sha256)
    }
}
