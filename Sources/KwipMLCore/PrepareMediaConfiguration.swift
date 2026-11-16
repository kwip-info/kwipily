import Foundation

/// Same catalog as the worker; editor drafts do not grant access or execute media.
public enum PrepareMediaConfiguration {
    public static let operatorID="kwip.data.prepareMedia"
    public struct Catalog:Codable {
        public let schemaVersion:Int
        public let defaults:[String:JSONValue]
        public let choices:[String:[String]]
        public let integerBounds:[String:[Int]]
    }
    private static let stored:Result<Catalog,Error>=Result {
        guard let url=Bundle.module.url(forResource:"prepare-media-options",withExtension:"json") else {throw failure("Media preparation options are unavailable.")}
        let bytes=try Data(contentsOf:url);guard bytes.count<=32768 else {throw failure("Media option catalog exceeds its allowance.")}
        let value=try JSONDecoder().decode(Catalog.self,from:bytes)
        guard value.schemaVersion==1 else {throw failure("Unsupported media preparation option catalog.")}
        return value
    }
    public static func catalog()throws->Catalog {try stored.get()}
    public static func defaults(kind:String="image")throws->[String:JSONValue] {
        var values=try catalog().defaults;values["kind"] = .string(kind)
        if kind=="audio" {values["inputColumn"] = .string("audio_content")}
        return values
    }
    public static func validate(_ values:[String:JSONValue])throws {
        let spec=try catalog()
        let optional:Set<String>=Set(["representation", "fftSize", "spectralHop", "window", "center", "melBands", "minimumHz", "maximumHz", "logFloorDB"]).union(["outputMode","cropEnabled","cropX","cropY","cropWidth","cropHeight","rotation"])
        let c=spec.defaults.merging(values){_,new in new}
        guard Set(values.keys).isSubset(of:Set(spec.defaults.keys)),Set(spec.defaults.keys).subtracting(optional).isSubset(of:Set(values.keys)) else {throw failure("Prepare media requires its complete versioned settings.")}
        for (key,choices) in spec.choices {
            guard case .string(let text)?=c[key],choices.contains(text) else {throw failure("Choose a supported "+key+".")}
        }
        for (key,bounds) in spec.integerBounds {
            guard case .integer(let number)?=c[key],bounds.count==2,number>=bounds[0],number<=bounds[1] else {throw failure("Media "+key+" exceeds its bound.")}
        }
        guard case .bool?=c["center"],let fft=integer(c["fftSize"]),let hop=integer(c["spectralHop"]),hop<=fft,c["representation"] == .string("samples") || c["kind"] == .string("audio") && c["sampleFormat"] == .string("float32") else {throw failure("Spectral output requires Audio/float32 and a valid hop.")}
        guard case .bool?=c["cropEnabled"],let rotation=integer(c["rotation"]),[0,90,180,270].contains(rotation) else {throw failure("Choose a valid crop and quarter-turn rotation.")}
        if c["kind"] == .string("image"),c["outputMode"] == .string("image") {guard c["sampleFormat"] == .string("preserve"),c["scaling"] == .string("none") else {throw failure("Image output preserves pixels; use tensor output for scaling.")}}
        for key in ["inputColumn","outputColumn"] {
            guard case .string(let name)?=c[key],!name.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,name.count<=200,!name.contains("\0") else {throw failure("Choose a bounded nonblank media column name.")}
        }
        guard c["inputColumn"] != c["outputColumn"],case .array(let matte)?=c["matte"],matte.count==3,
              matte.allSatisfy({value in switch value {case .integer(let n):return (0...1).contains(n);case .number(let n):return n.isFinite && (0...1).contains(n);default:return false}}),
              let width=integer(c["width"]),let height=integer(c["height"]),width*height<=16_000_000 else {throw failure("Invalid output name, composite matte or image pixel allowance.")}
        if c["kind"] == .string("audio") {
            guard c["scaling"] == .string("none"),c["sizeMode"] == .string("preserve"),c["color"] == .string("preserve"),c["alpha"] == .string("preserve"),
                  let window=integer(c["windowSamples"]),let hop=integer(c["hopSamples"]),window>0 || hop==0,window==0 || hop==0 || hop<=window else {throw failure("Audio preparation requires explicit valid windows and audio-only settings.")}
        }
    }
    private static func integer(_ value:JSONValue?)->Int64? {if case .integer(let n)=value {return n};return nil}
    private static func failure(_ message:String)->DataFlowContractError {.init(message)}
}
