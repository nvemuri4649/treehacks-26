// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "GlazeGuard",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(
            name: "GlazeGuard",
            targets: ["GlazeGuard"]
        )
    ],
    targets: [
        .executableTarget(
            name: "GlazeGuard",
            path: "GlazeGuard"
        )
    ]
)
