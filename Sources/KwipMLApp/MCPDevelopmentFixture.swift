#if DEBUG
import Foundation
import KwipMLCore

/// Explicit disposable native-UAT bundle only. Normal development and release
/// bundles do not contain this Info.plist key; release compiles none of this code.
/// No existing projects, Keychain clients or production credentials are borrowed.
enum MCPDevelopmentFixture {
    static var directory:URL? {
        guard let raw=Bundle.main.object(forInfoDictionaryKey:"KwipMLMCPFixtureDirectory") as? String,
              let project=Bundle.main.url(forResource:"ProjectRoot",withExtension:"txt").flatMap({try? String(contentsOf:$0,encoding:.utf8).trimmingCharacters(in:.whitespacesAndNewlines)}) else{return nil}
        let root=URL(fileURLWithPath:project).appendingPathComponent(".local"),url=URL(fileURLWithPath:raw).standardizedFileURL
        guard url.deletingLastPathComponent()==root.standardizedFileURL,url.lastPathComponent.hasPrefix("mcp-native-fixture-"),
              url.resolvingSymlinksInPath()==url else{return nil}
        return url
    }
    struct Secrets:MCPSecretStorage {
        let root:URL
        func read(clientID:String)throws->String? {
            let bytes=try DiagnosticFileIO.read(root:root,parts:["fixture-credentials.json"],limit:8192)
            return try JSONDecoder().decode([String:String].self,from:bytes)[clientID]
        }
        func write(_ token:String,clientID:String)throws{throw MCPSecretFailure.unavailable}
        func remove(clientID:String)throws{throw MCPSecretFailure.unavailable}
    }
}
#endif
