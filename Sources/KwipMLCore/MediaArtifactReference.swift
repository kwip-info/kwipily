import Foundation
import CryptoKit

/// A path-free immutable media pin. Only verified run inventory can resolve it.
public struct MediaArtifactReference:Codable,Equatable,Hashable {
    public let schemaVersion:Int
    public let kind:String
    public let sha256:String
    public let shape:[Int]
    public let axes:[String]
    public let dtype:String
    public let metadata:[String:JSONValue]
    public static let maximumArtifactBytes=50_000_000
    public static let maximumHeaderBytes=8192
    private static let magic=Data("KWIPMEDIA1\n".utf8)
    private static let keys:Set<String>=["schemaVersion","kind","sha256","shape","axes","dtype","metadata"]
    private static let sizes=["uint8":1,"uint16":2,"int16":2,"int32":4,"float32":4,"int64":8]
    public var artifactKind:String {kind == "audioSamplesRef" ? "audioSamples":"preparedTensor"}
    public var filename:String {"media-"+sha256+".kwipmedia"}
    public var byteCount:Int {shape.reduce(Self.sizes[dtype] ?? 0,*)}
    public static func decode(_ bytes:Data)throws->Self {
        guard bytes.count<=maximumHeaderBytes else {throw failure("Media reference exceeds its byte allowance.")}
        return try JSONDecoder().decode(Self.self,from:bytes)
    }
    public init(from decoder:Decoder)throws {
        let value=try decoder.singleValueContainer().decode(JSONValue.self)
        guard case .object(let fields)=value,Set(fields.keys)==Self.keys,fields["schemaVersion"] == 1,
              case .string(let kind)?=fields["kind"],["audioSamplesRef","preparedTensorRef"].contains(kind),
              case .string(let pin)?=fields["sha256"],Self.pin(pin),
              case .array(let dimensions)?=fields["shape"],(1...8).contains(dimensions.count),
              case .array(let names)?=fields["axes"],names.count==dimensions.count,
              case .string(let dtype)?=fields["dtype"],var count=Self.sizes[dtype],
              case .object(let metadata)?=fields["metadata"] else {throw Self.failure("Invalid immutable media reference.")}
        var shape:[Int]=[];var axes:[String]=[]
        for (dimension,name) in zip(dimensions,names) {
            guard case .integer(let raw)=dimension,let n=Int(exactly:raw),n>0,n<=Self.maximumArtifactBytes/count,
                  case .string(let axis)=name,["samples","channels","height","width","features","tokens","frequency","time","frames"].contains(axis),!axes.contains(axis) else {throw Self.failure("Invalid media shape, axes or byte allowance.")}
            count*=n;shape.append(n);axes.append(axis)
        }
        if kind == "audioSamplesRef" {
            guard Set(metadata.keys)==["sampleRate","sourceBits"],case .integer(let rate)?=metadata["sampleRate"],(1...768000).contains(rate),
                  case .integer(let bits)?=metadata["sourceBits"],[8:"uint8",16:"int16",24:"int32",32:"int32"][bits]==dtype,
                  axes == ["samples","channels"] else {throw Self.failure("Audio must preserve integer PCM precision, rate and channels.")}
        } else {
            func frequency(_ value:JSONValue?)->Double? {switch value {case .integer(let n):return Double(n);case .number(let n):return n.isFinite ? n:nil;default:return nil}}
            let allowed:Set<String>=["parentSHA256","recipeSHA256","sourceStart","sourceEnd","sourceRate","validSamples","sampleRate","sourceBits","geometry","maskConvention","spectral","validFrames"]
            guard Set(metadata.keys).isSubset(of:allowed) else {throw Self.failure("Unsupported prepared media metadata.")}
            for (key,value) in metadata {
                if key.hasSuffix("SHA256") {
                    guard case .string(let text)=value,Self.pin(text) else {throw Self.failure("Invalid prepared media provenance pin.")}
                } else if key == "sourceBits" {
                    guard case .integer(let bits)=value,[8,16,24,32].contains(bits),
                          (axes == ["channels","samples"] && [8:"uint8",16:"int16",24:"int32",32:"int32"][bits]==dtype || axes == ["samples"] && dtype=="int64" && metadata["maskConvention"] != nil) else {throw Self.failure("Prepared PCM precision differs from its waveform or mask.")}
                } else if key == "spectral" {
                    guard case .object(let spec)=value,Set(spec.keys)==["representation","fftSize","frameHopSamples","frameOffsetSamples","window","center","melBands","minimumHz","maximumHz","logFloorDB","melScale","melNormalization","logReference"],let fft=spec["fftSize"]?.integerValue,(16...4096).contains(fft),let hop=spec["frameHopSamples"]?.integerValue,(1...fft).contains(hop),case .bool(let center)?=spec["center"],spec["frameOffsetSamples"] == .integer(Int64(center ? -(fft/2):0)),let bands=spec["melBands"]?.integerValue,(8...256).contains(bands),let low=frequency(spec["minimumHz"]),let high=frequency(spec["maximumHz"]),0<=low,low<high,high<=384000,let floor=spec["logFloorDB"]?.integerValue,(-120 ... -1).contains(floor),["magnitude","power","logMel"].contains(spec["representation"]?.stringValue ?? ""),["hann","hamming","rectangular"].contains(spec["window"]?.stringValue ?? ""),spec["melScale"] == .string("htk"),spec["melNormalization"] == .string("none"),spec["logReference"] == .string("unitPower"),(axes == ["frames"] && dtype=="int64" || axes == ["channels","frames","frequency"] && dtype=="float32") else {throw Self.failure("Invalid spectral descriptor/axes.")}
                } else if key == "maskConvention" {
                    guard case .string(let convention)=value,["valid","blocked"].contains(convention) else {throw Self.failure("Invalid prepared mask convention.")}
                } else if key == "geometry" {
                    guard case .array(let matrix)=value,matrix.count==9,matrix.allSatisfy({v in switch v {case .integer:return true;case .number(let x):return x.isFinite;default:return false}}) else {throw Self.failure("Invalid image geometry transform.")}
                } else {
                    guard case .integer(let number)=value,(0...9_007_199_254_740_991).contains(number) else {throw Self.failure("Invalid media sample metadata.")}
                }
            }
            if metadata["sourceStart"] != nil || metadata["sourceEnd"] != nil {
                guard case .integer(let start)?=metadata["sourceStart"],case .integer(let end)?=metadata["sourceEnd"],end>=start,
                      case .integer(let rate)?=metadata["sourceRate"],rate>0 else {throw Self.failure("Invalid source sample interval.")}
            }
            if case .object(let spec)?=metadata["spectral"] {
                guard let rate=metadata["sampleRate"]?.integerValue,rate>0,let high=frequency(spec["maximumHz"]),high<=Double(rate)/2 else {throw Self.failure("Spectral frequencies exceed their sample rate.")}
                if axes == ["channels","frames","frequency"] {guard let fft=spec["fftSize"]?.integerValue,let bands=spec["melBands"]?.integerValue,shape[2] == (spec["representation"] == .string("logMel") ? bands:fft/2+1) else {throw Self.failure("Spectral bins differ from their recipe.")}}
            }
            if let valid=metadata["validFrames"] {
                guard case .integer(let n)=valid,let i=axes.firstIndex(of:"frames"),n<=shape[i],metadata["spectral"] != nil else {throw Self.failure("Valid frames exceed the spectral extent.")}
            }
            if let valid=metadata["validSamples"] {
                guard case .integer(let n)=valid,let i=axes.firstIndex(of:"samples"),n<=shape[i] else {throw Self.failure("Valid samples exceed the tensor extent.")}
            }
        }
        schemaVersion=1;self.kind=kind;sha256=pin;self.shape=shape;self.axes=axes;self.dtype=dtype;self.metadata=metadata
    }
    public struct Decoded {public let reference:MediaArtifactReference;public let samples:Data}
    public static func decode(container:Data,expected:Self)throws->Decoded {
        let start=magic.count+4
        guard container.count>start,container.count<=maximumArtifactBytes,container.starts(with:magic),
              SHA256.hash(data:container).map({String(format:"%02x",$0)}).joined()==expected.sha256 else {throw failure("Media container differs from its immutable pin.")}
        let length=container[magic.count..<start].reduce(0){($0<<8)|Int($1)}
        guard (1...maximumHeaderBytes).contains(length),start+length<=container.count else {throw failure("Invalid media container header size.")}
        let bytes=container.subdata(in:start..<start+length)
        guard bytes.allSatisfy({$0<128}),case .object(var fields)=try JSONDecoder().decode(JSONValue.self,from:bytes),
              Set(fields.keys)==["schemaVersion","kind","shape","axes","dtype","metadata","byteOrder"],
              fields["kind"] == .string(expected.artifactKind),fields["byteOrder"] == .string("little") else {throw failure("Unsupported media container.")}
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys,.withoutEscapingSlashes]
        guard try encoder.encode(JSONValue.object(fields))==bytes else {throw failure("Media header is not canonical.")}
        fields.removeValue(forKey:"byteOrder");fields["kind"] = .string(expected.kind);fields["sha256"] = .string(expected.sha256)
        guard try decode(encoder.encode(JSONValue.object(fields)))==expected,container.count-start-length==expected.byteCount else {throw failure("Media geometry differs from its typed reference.")}
        let samples=container.subdata(in:start+length..<container.count)
        if expected.dtype == "float32" {
            let finite=try samples.withUnsafeBytes {raw in
                for index in stride(from:0,to:raw.count,by:4) {
                    if index%262144==0 {try Task.checkCancellation()}
                    if !Float(bitPattern:UInt32(littleEndian:raw.loadUnaligned(fromByteOffset:index,as:UInt32.self))).isFinite {return false}
                }
                return true
            }
            guard finite else {throw failure("Prepared media samples must be finite.")}
        }
        return .init(reference:expected,samples:samples)
    }
    private static func pin(_ value:String)->Bool {value.utf8.count==64 && value.utf8.allSatisfy{(48...57).contains($0)||(97...102).contains($0)}}
    private static func failure(_ message:String)->DataFlowContractError {.init(message)}
}
