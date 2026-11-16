import Foundation
import KwipMLCore

struct DirectUpdateOffer {
    let version:String
    let build:String
    let url:URL?
    let size:UInt64
    var signedFeed:Bool
    var isDelta=false
    var deltaFrom:String?=nil
    var artifactSignature:String?=nil
}

/// A cycle's validated target; fallback may change artifact but never target release.
struct DirectUpdateTarget:Equatable {
    let version:String
    let build:String
    init?(offer:DirectUpdateOffer,installedBuild:String) {
        guard DirectUpdatePolicy.accepts(offer,installedBuild:installedBuild) else{return nil}
        version=offer.version;build=offer.build
    }
}

/// One admission state shared by the actual Sparkle callbacks and focused lifecycle tests.
struct DirectUpdateCycleAdmission {
    private(set) var target:DirectUpdateTarget?
    private(set) var metadataValid=true
    mutating func reset() {target=nil;metadataValid=true}
    mutating func observe(_ parents:[[AnyHashable:Any]]) {
        guard metadataValid else{return}
        if !parents.allSatisfy(DirectUpdatePolicy.deltaMetadataValid) {metadataValid=false;target=nil}
    }
    mutating func admit(_ offer:DirectUpdateOffer,installedBuild:String,activeWork:Bool)->Bool {
        guard metadataValid,!activeWork,let next=DirectUpdateTarget(offer:offer,installedBuild:installedBuild),
              target == nil || target == next else{return false}
        target=next;return true
    }
}

enum DirectUpdatePolicy {
    static let feedURL=DesktopDeployment.origin+"/updates/v1/arm64/appcast.xml"
    static let publicKey="maob1E5oufy9KNFYZjxnd7/rEb/Ld13LqW6KFmc3aVc="
    static func accepts(_ offer:DirectUpdateOffer,installedBuild:String)->Bool {
        func build(_ value:String)->UInt64? {
            guard value.range(of:#"\A[1-9][0-9]{0,8}\z"#,options:.regularExpression) != nil else{return nil}
            return UInt64(value)
        }
        guard offer.signedFeed,offer.version.range(of:#"\A1\.(0|[1-9][0-9]{0,3})\.(0|[1-9][0-9]{0,3})\z"#,options:.regularExpression) != nil,
              let offered=build(offer.build),let installed=build(installedBuild),offered>installed,
              offer.size>0,offer.size<=5*1024*1024*1024,let url=offer.url else{return false}
        let prefix=DesktopDeployment.origin+"/downloads/\(offer.build)/"
        guard url.absoluteString.hasPrefix(prefix) else{return false}
        let suffix=String(url.absoluteString.dropFirst(prefix.count))
        if offer.isDelta {
            guard let base=offer.deltaFrom,base==installedBuild,build(base) != nil,
                  let signature=offer.artifactSignature,
                  let bytes=Data(base64Encoded:signature),bytes.count==64,
                  bytes.base64EncodedString()==signature else{return false}
            return suffix.range(of:#"\A[a-f0-9]{64}/KwipML-from-"#+base+#"\.delta\z"#,options:.regularExpression) != nil
        }
        guard offer.deltaFrom == nil else{return false}
        return suffix.range(of:#"\A[a-f0-9]{64}/KwipML\.dmg\z"#,options:.regularExpression) != nil
    }
    /// Sparkle can reuse captured headers for fallback downloads. Revalidate each concrete request.
    static func prepareDownloadRequest(_ request:NSMutableURLRequest,offer:DirectUpdateOffer,
                                      installedBuild:String)->Bool {
        request.setValue(nil,forHTTPHeaderField:"Authorization")
        request.setValue(nil,forHTTPHeaderField:"Cookie")
        request.httpShouldHandleCookies=false
        guard accepts(offer,installedBuild:installedBuild),
              request.url?.absoluteString==offer.url?.absoluteString else {
            // This callback cannot throw. An unsupported scheme prevents any network request.
            request.url=URL(string:"about:blank")
            return false
        }
        return true
    }
    static func prepareDownloadRequest(_ request:NSMutableURLRequest,offer:DirectUpdateOffer,
                                      installedBuild:String,target:DirectUpdateTarget,activeWork:Bool)->Bool {
        request.setValue(nil,forHTTPHeaderField:"Authorization")
        request.setValue(nil,forHTTPHeaderField:"Cookie")
        request.httpShouldHandleCookies=false
        guard !activeWork,offer.version==target.version,offer.build==target.build else {
            request.url=URL(string:"about:blank");return false
        }
        return prepareDownloadRequest(request,offer:offer,installedBuild:installedBuild)
    }
    /// Raw parent metadata retains duplicate base entries that Sparkle's map collapses.
    static func deltaMetadataValid(_ properties:[AnyHashable:Any])->Bool {
        guard let raw=properties["sparkle:deltas"] else{return true}
        guard let deltas=raw as? [[String:Any]],deltas.count<=16 else{return false}
        var bases=Set<String>()
        for delta in deltas {
            guard let base=delta["sparkle:deltaFrom"] as? String,
                  base.range(of:#"\A[1-9][0-9]{0,8}\z"#,options:.regularExpression) != nil,
                  bases.insert(base).inserted else{return false}
        }
        return true
    }
    static func canCheck(production:Bool,configured:Bool,activeWork:Bool)->Bool {
        production && configured && !activeWork
    }
    static func configured(_ info:[String:Any])->Bool {
        guard DesktopDeployment.isCommercialEdition(bundle:info["CFBundleIdentifier"] as? String),
              info["KWIPMLDirectUpdatesEnabled"] as? Bool==true,
              info["SUFeedURL"] as? String==feedURL,info["SUPublicEDKey"] as? String==publicKey,
              info["SURequireSignedFeed"] as? Bool==true,
              info["SUVerifyUpdateBeforeExtraction"] as? Bool==true,
              let expiry=info["SUSignedFeedFailureExpirationInterval"] as? NSNumber,expiry.doubleValue==0,
              info["SUAllowsAutomaticUpdates"] as? Bool==false,info["SUAutomaticallyUpdate"] as? Bool==false else{return false}
        return ["SUEnableDownloaderService","SUEnableInstallerLauncherService","SUEnableInstallerConnectionService","SUEnableInstallerStatusService"].allSatisfy{info[$0] as? Bool==true}
    }
}
