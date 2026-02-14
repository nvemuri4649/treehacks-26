#!/bin/bash

# Create Xcode project for GlazeGuard
set -e

echo "🔧 Creating Xcode project for GlazeGuard..."

# Create Xcode project directory structure
PROJECT_DIR="GlazeGuard.xcodeproj"
mkdir -p "$PROJECT_DIR"

# Get all Swift source files
SOURCES=$(find GlazeGuard -name "*.swift" | sed 's/^/\t\t\t\t/' | sed 's/$/,/')

# Create project.pbxproj
cat > "$PROJECT_DIR/project.pbxproj" << 'EOF'
// !$*UTF8*$!
{
	archiveVersion = 1;
	classes = {
	};
	objectVersion = 56;
	objects = {
		MAIN_GROUP /* Project object */ = {
			isa = PBXProject;
			attributes = {
				BuildIndependentTargetsInParallel = 1;
				LastSwiftUpdateCheck = 1500;
				LastUpgradeCheck = 1500;
			};
			buildConfigurationList = BUILD_CONFIGURATION_LIST;
			compatibilityVersion = "Xcode 14.0";
			developmentRegion = en;
			hasScannedForEncodings = 0;
			knownRegions = (
				en,
				Base,
			);
			mainGroup = MAIN_GROUP_REF;
			productRefGroup = PRODUCTS_GROUP;
			projectDirPath = "";
			projectRoot = "";
			targets = (
				TARGET_GLAZEGUARD,
			);
		};
	};
	rootObject = MAIN_GROUP;
}
EOF

echo "✅ Xcode project structure created"
echo ""
echo "⚠️  For macOS apps with menu bars, it's best to use Xcode directly:"
echo ""
echo "  cd GlazeGuard"
echo "  open -a Xcode ."
echo ""
echo "Then in Xcode:"
echo "  1. File → New → Project"
echo "  2. Choose 'macOS → App'"
echo "  3. Name: GlazeGuard"
echo "  4. Interface: SwiftUI, Lifecycle: SwiftUI App"
echo "  5. Replace the generated files with our Swift files"
echo ""
echo "Or try the simpler command-line build:"
echo "  swift build"
echo ""

exit 0
EOF

chmod +x create_xcode_project.sh
