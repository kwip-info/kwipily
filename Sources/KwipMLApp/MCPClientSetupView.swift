import SwiftUI
import AppKit
import KwipMLCore

/// Display masking is separate from the exact payload copied to the client.
struct MCPConnectionDetails {
    let configuration:String
    let endpoint:String
    let authorization:String
    private let maskedConfiguration:String
    init?(configuration:String) {
        guard let data=configuration.data(using:.utf8),
              case .object(var root)=try? JSONDecoder().decode(JSONValue.self,from:data),
              case .object(var servers)=root["mcpServers"],case .object(var client)=servers["kwip-ml"],
              let endpoint=client["url"]?.stringValue,let url=URLComponents(string:endpoint),
              url.scheme=="http",url.host=="127.0.0.1",url.path=="/mcp",url.query==nil,url.fragment==nil,
              url.user==nil,url.password==nil,let port=url.port,(1024...65535).contains(port),
              case .object(var headers)=client["headers"],let authorization=headers["Authorization"]?.stringValue,
              authorization.hasPrefix("Bearer "),authorization.count>7 else{return nil}
        self.configuration=configuration;self.endpoint=endpoint;self.authorization=authorization
        headers["Authorization"] = .string("Bearer <client credential>");client["headers"] = .object(headers)
        servers["kwip-ml"] = .object(client);root["mcpServers"] = .object(servers)
        let encoder=JSONEncoder();encoder.outputFormatting=[.prettyPrinted,.sortedKeys,.withoutEscapingSlashes]
        guard let masked=try? encoder.encode(JSONValue.object(root))else{return nil}
        maskedConfiguration=String(decoding:masked,as:UTF8.self)
    }
    func displayConfiguration(reveal:Bool)->String {reveal ? configuration:maskedConfiguration}
}

struct MCPClientSetupView:View {
    let clientName:String
    let configuration:String
    let onDone:()->Void
    @State private var revealCredential=false
    @Environment(\.appearanceTheme) private var theme
    private var details:MCPConnectionDetails? {MCPConnectionDetails(configuration:configuration)}
    var body:some View {
        VStack(alignment:.leading,spacing:18) {
            HStack(spacing:12) {
                Image(systemName:"point.3.connected.trianglepath.dotted").font(.system(size:22))
                    .foregroundStyle(theme.accentColor).frame(width:44,height:44)
                    .background(theme.accentColor.opacity(0.10),in:RoundedRectangle(cornerRadius:8))
                VStack(alignment:.leading,spacing:4) {
                    Text("Connect a client").font(.system(size:22,weight:.semibold))
                    Text(clientName).font(.callout).foregroundStyle(MLTheme.muted).lineLimit(1)
                }
                Spacer()
                Text("Local HTTP").font(.caption).foregroundStyle(theme.accentColor)
            }
            Text("Add these details in your AI client’s MCP settings.").font(.callout).foregroundStyle(MLTheme.muted)
            if let details {
                ThemedScrollView {
                    VStack(alignment:.leading,spacing:18) {
                field("Server URL",value:details.endpoint,copy:details.endpoint)
                VStack(alignment:.leading,spacing:10) {
                    HStack {
                        Text("Authorization header").font(.system(size:13,weight:.semibold))
                        Spacer()
                        Button {revealCredential.toggle()} label:{Image(systemName:revealCredential ? "eye.slash":"eye")}
                            .help(revealCredential ? "Hide credential":"Show credential")
                            .accessibilityLabel(revealCredential ? "Hide credential":"Show credential")
                        Button("Copy header"){copy(details.authorization)}
                    }
                    Text(revealCredential ? details.authorization:"Bearer ••••••••••••••••••••••••")
                        .font(.system(.callout,design:.monospaced)).textSelection(.enabled)
                        .fixedSize(horizontal:false,vertical:true)
                    Text("Private credential · keep it out of recordings and chat.").font(.caption).foregroundStyle(MLTheme.muted)
                }.padding(14).frame(maxWidth:.infinity,alignment:.leading).background(MLTheme.panel,in:RoundedRectangle(cornerRadius:8))
                    .overlay(RoundedRectangle(cornerRadius:8).stroke(MLTheme.rule))
                DisclosureGroup("JSON configuration") {
                    VStack(alignment:.leading,spacing:10) {
                        Text("Example for clients using mcpServers. Other harnesses may use a different format.")
                            .font(.caption).foregroundStyle(MLTheme.muted)
                        Text(details.displayConfiguration(reveal:revealCredential))
                            .font(.system(.caption,design:.monospaced)).textSelection(.enabled)
                            .frame(maxWidth:.infinity,alignment:.leading)
                    }.padding(.top,10)
                }
                    }.padding(.trailing,4)
                }.frame(height:300)
                Divider()
                HStack {
                    Button("Copy configuration"){copy(details.configuration)}
                    Spacer()
                    Button("Done",action:onDone).keyboardShortcut(.defaultAction)
                }
            }else{
                Text("Connection details are unavailable. Close this sheet and try again.").font(.callout)
                Button("Done",action:onDone)
            }
        }.padding(24).frame(width:600).background(MLTheme.background).buttonStyle(FlatButtonStyle())
    }
    private func field(_ title:String,value:String,copy payload:String)->some View {
        VStack(alignment:.leading,spacing:10) {
            HStack {Text(title).font(.system(size:13,weight:.semibold));Spacer();Button("Copy URL"){copy(payload)}}
            Text(value).font(.system(.callout,design:.monospaced)).textSelection(.enabled)
                .fixedSize(horizontal:false,vertical:true)
        }.padding(14).frame(maxWidth:.infinity,alignment:.leading).background(MLTheme.panel,in:RoundedRectangle(cornerRadius:8))
            .overlay(RoundedRectangle(cornerRadius:8).stroke(MLTheme.rule))
    }
    private func copy(_ text:String){NSPasteboard.general.clearContents();NSPasteboard.general.setString(text,forType:.string)}
}
