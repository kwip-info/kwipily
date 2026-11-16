import Foundation

/// The additive discovery declaration required before a worker may execute a GELU graph.
/// It does not broaden the historical product execution contract or any lesson allowlist.
public struct GeluRuntimeCapability:Decodable,Equatable {
    public let version:ContractVersion
    public let adapterID:String
    public let graphSchemaVersions:[Int]
    public let approximateTanhModes:[Bool]
    private struct Key:CodingKey {
        var stringValue:String
        var intValue:Int? {nil}
        init?(stringValue:String) {self.stringValue=stringValue}
        init?(intValue:Int) {return nil}
        init(_ value:String) {stringValue=value}
    }
    public init(from decoder:Decoder)throws {
        let c=try decoder.container(keyedBy:Key.self)
        guard Set(c.allKeys.map(\.stringValue)) == ["version","adapterID","graphSchemaVersions","approximateTanhModes"] else {
            throw GraphConfigurationError("GELU capability has unknown or missing fields.")
        }
        let revision=try c.nestedContainer(keyedBy:Key.self,forKey:Key("version"))
        guard Set(revision.allKeys.map(\.stringValue)) == ["major","minor","patch"] else {throw GraphConfigurationError("GELU capability version fields differ.")}
        version=ContractVersion(try revision.decode(Int.self,forKey:Key("major")),try revision.decode(Int.self,forKey:Key("minor")),try revision.decode(Int.self,forKey:Key("patch")))
        adapterID=try c.decode(String.self,forKey:Key("adapterID"))
        graphSchemaVersions=try c.decode([Int].self,forKey:Key("graphSchemaVersions"))
        approximateTanhModes=try c.decode([Bool].self,forKey:Key("approximateTanhModes"))
        guard version == ContractVersion(),adapterID == "kwip.runner.graph-v2.gelu",graphSchemaVersions == [2],approximateTanhModes == [false,true] else {throw GraphConfigurationError("GELU capability does not match the reviewed version, adapter, schema and approximation modes.")}
    }
    public static func decode(_ data:Data)throws->Self {
        guard data.count<=4096 else {throw GraphConfigurationError("GELU capability exceeds 4096 bytes.")}
        return try JSONDecoder().decode(Self.self,from:data)
    }
}
