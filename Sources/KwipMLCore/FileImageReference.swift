import Foundation
import CryptoKit

/// A content pin, never a path or a grant. Resolution belongs to verified owned inventory.
public struct FileImageReference:Codable,Equatable,Hashable,Sendable {
    public let schemaVersion:Int
    public let kind:String
    public let sha256:String
    public let width:Int
    public let height:Int
    public let channels:Int
    public let dtype:String
    public let layout:String
    public static let maximumArtifactBytes=50_000_000
    public static let maximumHeaderBytes=4096
    private static let magic=Data("KWIPIMG1\n".utf8)
    private static let keys:Set<String>=["schemaVersion","kind","sha256","width","height","channels","dtype","layout"]
    private static let sizes=["uint8":1,"uint16":2,"int32":4,"float32":4]
    private static let modes:[String:(Int,String)]=["L":(1,"uint8"),"LA":(2,"uint8"),"RGB":(3,"uint8"),"RGBA":(4,"uint8"),"CMYK":(4,"uint8"),"I;16":(1,"uint16"),"I":(1,"int32"),"F":(1,"float32")]
    public init(sha256:String,width:Int,height:Int,channels:Int,dtype:String,layout:String)throws {
        guard sha256.utf8.count==64,sha256.utf8.allSatisfy({(48...57).contains($0) || (97...102).contains($0)}) else {throw Self.failure("Invalid image artifact SHA-256 pin.")}
        _=try Self.geometry(width:width,height:height,channels:channels,dtype:dtype,layout:layout)
        schemaVersion=1;kind="imagePixelsRef";self.sha256=sha256;self.width=width;self.height=height;self.channels=channels;self.dtype=dtype;self.layout=layout
    }
    public init(from decoder:Decoder)throws {
        let value=try decoder.singleValueContainer().decode(JSONValue.self)
        guard case .object(let fields)=value,Set(fields.keys)==Self.keys,fields["schemaVersion"] == 1,fields["kind"] == .string("imagePixelsRef"),
              case .string(let sha)?=fields["sha256"],case .integer(let width)?=fields["width"],case .integer(let height)?=fields["height"],case .integer(let channels)?=fields["channels"],
              let w=Int(exactly:width),let h=Int(exactly:height),let c=Int(exactly:channels),case .string(let dtype)?=fields["dtype"],case .string(let layout)?=fields["layout"] else {throw Self.failure("Invalid immutable image reference.")}
        try self.init(sha256:sha,width:w,height:h,channels:c,dtype:dtype,layout:layout)
    }
    public static func decode(_ data:Data)throws->Self {
        guard data.count<=4096 else {throw failure("Image reference exceeds its byte bound.")}
        return try JSONDecoder().decode(Self.self,from:data)
    }
    public struct Decoded:Sendable {
        public let reference:FileImageReference
        public let pixels:Data
        public let policy:[String:String]
    }
    public static func decode(container:Data,expected:FileImageReference)throws->Decoded {
        let start=magic.count+4
        guard container.count>start,container.count<=maximumArtifactBytes,container.starts(with:magic),
              SHA256.hash(data:container).map({String(format:"%02x",$0)}).joined()==expected.sha256 else {throw failure("Image container differs from its immutable artifact pin.")}
        let length=container[magic.count..<start].reduce(0) {($0<<8)|Int($1)}
        guard (1...maximumHeaderBytes).contains(length),start+length<=container.count else {throw failure("Image container header exceeds its byte bound.")}
        let bytes=container.subdata(in:start..<start+length)
        guard bytes.allSatisfy({$0<128}),case .object(let header)=try JSONDecoder().decode(JSONValue.self,from:bytes),
              Set(header.keys)==["schemaVersion","kind","width","height","channels","dtype","layout","byteOrder","policy"],header["schemaVersion"] == 1,header["kind"] == .string("imagePixels"),header["byteOrder"] == .string("little"),
              header["width"] == .integer(Int64(expected.width)),header["height"] == .integer(Int64(expected.height)),header["channels"] == .integer(Int64(expected.channels)),header["dtype"] == .string(expected.dtype),header["layout"] == .string(expected.layout),
              case .object(let rawPolicy)?=header["policy"],Set(rawPolicy.keys)==["orientation","colorMode","palette","actualMode"] else {throw failure("Image container header differs from its typed reference.")}
        var policy:[String:String]=[:]
        for (key,value) in rawPolicy {guard case .string(let text)=value else {throw failure("Invalid image decoding policy.")};policy[key]=text}
        guard ["stored","applyEXIF"].contains(policy["orientation"] ?? ""),["preserve","RGB","RGBA","L"].contains(policy["colorMode"] ?? ""),["expandRGBA","reject"].contains(policy["palette"] ?? ""),
              let mode=modes[policy["actualMode"] ?? ""],mode.0==expected.channels,mode.1==expected.dtype else {throw failure("Invalid image decoding policy or actual mode.")}
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys,.withoutEscapingSlashes]
        guard try encoder.encode(JSONValue.object(header))==bytes else {throw failure("Image container header is not canonical.")}
        let size=try geometry(width:expected.width,height:expected.height,channels:expected.channels,dtype:expected.dtype,layout:expected.layout)
        guard container.count-start-length==size else {throw failure("Image pixel bytes differ from declared geometry.")}
        return .init(reference:expected,pixels:container.subdata(in:start+length..<container.count),policy:policy)
    }
    private static func geometry(width:Int,height:Int,channels:Int,dtype:String,layout:String)throws->Int {
        guard (1...16384).contains(width),(1...16384).contains(height),(1...4).contains(channels),width*height<=16_000_000,let size=sizes[dtype],["HWC","CHW"].contains(layout) else {throw failure("Image geometry or dtype is outside its bounded contract.")}
        let bytes=width*height*channels*size
        guard bytes<=maximumArtifactBytes else {throw failure("Decoded image exceeds its 50 MB artifact allowance.")}
        return bytes
    }
    private static func failure(_ message:String)->DataFlowContractError {.init(message)}
}
